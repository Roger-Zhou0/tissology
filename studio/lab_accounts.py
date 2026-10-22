"""Single-lab membership. Credentials live outside scientific/import directories."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import threading
import time

SCRYPT_N = 2**17
SESSION_SECONDS = 12 * 3600
INVITE_SECONDS = 7 * 86400
_PASSWORD_SLOTS = threading.BoundedSemaphore(2)


class LabError(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def token_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def email_key(value):
    value = value.strip().lower()
    if len(value) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
        raise LabError('Enter a valid email address.', 422)
    return value


def display_name(value, label='Name'):
    value = value.strip()
    if not 1 <= len(value) <= 120 or any(ord(c) < 32 for c in value):
        raise LabError(f'{label} must contain 1–120 printable characters.', 422)
    return value


def _password_digest(password, *, salt, n, r=8, p=1):
    # One scrypt operation consumes about 128 MiB. Bound simultaneous work so
    # a login burst cannot exhaust the CPU server's memory alongside encoders.
    if not _PASSWORD_SLOTS.acquire(timeout=2):
        raise LabError('Sign-in is busy. Try again shortly.', 429)
    try:
        return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=n, r=r, p=p,
                              maxmem=256*1024*1024).hex()
    finally:
        _PASSWORD_SLOTS.release()


def password_hash(password, salt=None):
    if not 12 <= len(password) <= 256:
        raise LabError('Use a password with 12–256 characters.', 422)
    salt = salt or secrets.token_hex(16)
    digest = _password_digest(password, salt=bytes.fromhex(salt), n=SCRYPT_N)
    return f'scrypt${SCRYPT_N}$8$1${salt}${digest}'


def password_matches(password, encoded):
    if not 12 <= len(password) <= 256:
        return False
    try:
        algorithm, n, r, p, salt, expected = encoded.split('$')
        # Only our recorded scheme is supported. Corrupt metadata must not turn
        # login into an unbounded expensive operation or expose a traceback.
        n, r, p = int(n), int(r), int(p)
        if (algorithm != 'scrypt' or n < 2 or n > 2**17 or n & (n-1)
                or r != 8 or p != 1 or not re.fullmatch(r'[a-f0-9]{32}', salt)
                or not re.fullmatch(r'[a-f0-9]{128}', expected)):
            return False
        actual = _password_digest(password, salt=bytes.fromhex(salt), n=n, r=r, p=p)
    except (AttributeError, TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


class LabAccounts:
    def __init__(self, directory, *, now=time.time):
        self.directory = Path(directory).absolute()
        if self.directory.is_symlink():
            raise ValueError('Lab credential directory cannot be a symbolic link')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory_stat = self.directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode) or directory_stat.st_uid != os.geteuid():
            raise ValueError('Lab credential directory must be owned by the deployment user')
        self.directory.chmod(0o700)
        self.path = self.directory / 'lab.sqlite3'
        self.setup_path = self.directory / 'owner-setup.txt'
        for path in (self.path, self.setup_path):
            self._check_file(path)
        self.now = now
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS members (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT NOT NULL,
                  active INTEGER NOT NULL DEFAULT 1, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, member_id TEXT NOT NULL,
                  csrf TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS invitations (id TEXT PRIMARY KEY, email TEXT NOT NULL,
                  token TEXT UNIQUE NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                  consumed REAL, revoked REAL);
                CREATE TABLE IF NOT EXISTS attempts (bucket TEXT PRIMARY KEY, started REAL NOT NULL, count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS activity (id TEXT PRIMARY KEY, actor_id TEXT NOT NULL,
                  actor_name TEXT NOT NULL, action TEXT NOT NULL, path TEXT NOT NULL,
                  status INTEGER, created REAL NOT NULL);
            ''')
            db.execute('BEGIN IMMEDIATE')
            if not db.execute("SELECT 1 FROM members WHERE role='owner' AND active=1").fetchone():
                # Exclusive file creation makes a crash between file/DB writes recoverable.
                if not self.setup_path.exists():
                    fd = os.open(self.setup_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'w') as out:
                        out.write(secrets.token_urlsafe(32) + '\n')
                self.setup_path.chmod(0o600)
                code = self.setup_path.read_text().strip()
                if not re.fullmatch(r'[A-Za-z0-9_-]{43}', code):
                    # Recover an interrupted first write without accepting an
                    # empty/truncated bootstrap credential.
                    code = secrets.token_urlsafe(32)
                    self.setup_path.write_text(code + '\n')
                db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', ('setup_hash', token_hash(code)))
            else:
                # A process can stop after the owner transaction commits but
                # before setup() removes this now-consumed secret.
                self.setup_path.unlink(missing_ok=True)
                db.execute("DELETE FROM settings WHERE key='setup_hash'")
            db.commit()
        self.path.chmod(0o600)

    @staticmethod
    def _check_file(path):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ValueError('Lab credential files must be regular files owned by the deployment user')

    @contextmanager
    def connect(self):
        self._check_file(self.path)
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def user(row):
        return {k: row[k] for k in ('id', 'name', 'email', 'role')} | {'created_at': stamp(row['created'])}

    def info(self):
        with self.connect() as db:
            ready = bool(db.execute("SELECT 1 FROM members WHERE active=1 AND role='owner'").fetchone())
            name = db.execute("SELECT value FROM settings WHERE key='name'").fetchone()
        return {'setup_required': not ready, 'workspace': {'name': name[0] if name else 'Shared lab'}}

    def throttle(self, bucket, limit=12, seconds=300):
        now = self.now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM attempts WHERE bucket=?', (bucket,)).fetchone()
            if row and row['started'] > now-seconds:
                if row['count'] >= limit:
                    raise LabError('Too many attempts. Try again in a few minutes.', 429)
                db.execute('UPDATE attempts SET count=count+1 WHERE bucket=?', (bucket,))
            else:
                db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,1)', (bucket, now))
            db.execute('DELETE FROM attempts WHERE started<?', (now-86400,))
            db.commit()

    def setup(self, code, name, email, password, workspace_name):
        self.throttle('setup', 10)
        name, email, workspace_name = display_name(name), email_key(email), display_name(workspace_name, 'Lab name')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            expected = db.execute("SELECT value FROM settings WHERE key='setup_hash'").fetchone()
            if db.execute("SELECT 1 FROM members WHERE role='owner'").fetchone():
                raise LabError('The lab has already been set up. Sign in instead.', 409)
            if not expected or not hmac.compare_digest(expected[0], token_hash(code)):
                raise LabError('The setup code is invalid.', 403)
            encoded = password_hash(password)
            identifier = secrets.token_hex(16)
            db.execute('INSERT INTO members VALUES (?,?,?,?,?,1,?)', (identifier,name,email,encoded,'owner',self.now()))
            db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', ('name',workspace_name))
            db.execute("DELETE FROM settings WHERE key='setup_hash'")
            db.commit()
            row = db.execute('SELECT * FROM members WHERE id=?', (identifier,)).fetchone()
        self.setup_path.unlink(missing_ok=True)
        return self.user(row)

    def login(self, email, password):
        email = email_key(email)
        self.throttle('login-global', 80)
        self.throttle('login:'+token_hash(email), 12)
        with self.connect() as db:
            row = db.execute('SELECT * FROM members WHERE email=? AND active=1', (email,)).fetchone()
        if row is None:
            # Match the expensive operation without retaining a dummy credential.
            if 12 <= len(password) <= 256:
                password_hash(password, '00'*16)
            raise LabError('Email or password is incorrect.', 401)
        if not password_matches(password, row['password']):
            raise LabError('Email or password is incorrect.', 401)
        return self.user(row)

    def create_session(self, member_id):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with self.connect() as db:
            if not db.execute('SELECT 1 FROM members WHERE id=? AND active=1', (member_id,)).fetchone():
                raise LabError('This membership is no longer active.', 401)
            db.execute('DELETE FROM sessions WHERE expires<?', (self.now(),))
            db.execute('INSERT INTO sessions VALUES (?,?,?,?)', (token_hash(token),member_id,csrf,self.now()+SESSION_SECONDS))
            db.commit()
        return token, csrf

    def session(self, token):
        if not token or len(token)>200:
            return None
        with self.connect() as db:
            row = db.execute('''SELECT m.*,s.csrf FROM sessions s JOIN members m ON s.member_id=m.id
                WHERE s.token=? AND s.expires>? AND m.active=1''', (token_hash(token),self.now())).fetchone()
        return {'user':self.user(row),'csrf_token':row['csrf']} if row else None

    def logout(self, token):
        with self.connect() as db:
            db.execute('DELETE FROM sessions WHERE token=?', (token_hash(token),))
            db.commit()

    @staticmethod
    def owner(db, actor_id):
        if not db.execute("SELECT 1 FROM members WHERE id=? AND role='owner' AND active=1", (actor_id,)).fetchone():
            raise LabError('Only the lab owner can manage membership.', 403)

    def create_invitation(self, actor_id, email):
        email = email_key(email)
        identifier, token, now = secrets.token_hex(16), secrets.token_urlsafe(32), self.now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self.owner(db, actor_id)
            if db.execute('SELECT 1 FROM members WHERE email=? AND active=1', (email,)).fetchone():
                raise LabError('This person is already a member.', 409)
            db.execute('UPDATE invitations SET revoked=? WHERE email=? AND consumed IS NULL AND revoked IS NULL', (now,email))
            db.execute('INSERT INTO invitations VALUES (?,?,?,?,?,NULL,NULL)', (identifier,email,token_hash(token),now,now+INVITE_SECONDS))
            db.commit()
        return {'id':identifier,'email':email,'token':token,'created_at':stamp(now),'expires_at':stamp(now+INVITE_SECONDS)}

    def join(self, token, name, email, password):
        self.throttle('join-global',30)
        name,email=display_name(name),email_key(email)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            invite=db.execute('''SELECT * FROM invitations WHERE token=? AND email=?
                AND consumed IS NULL AND revoked IS NULL AND expires>?''',(token_hash(token),email,self.now())).fetchone()
            if not invite:
                raise LabError('The invitation is invalid, expired, or belongs to another email address.',403)
            existing=db.execute('SELECT * FROM members WHERE email=?',(email,)).fetchone()
            if existing and existing['active']:
                raise LabError('This email already has an account. Sign in instead.',409)
            encoded=password_hash(password)
            identifier=existing['id'] if existing else secrets.token_hex(16)
            if existing:
                db.execute("UPDATE members SET name=?,password=?,active=1,role='member' WHERE id=?",(name,encoded,identifier))
                db.execute('DELETE FROM sessions WHERE member_id=?',(identifier,))
            else:
                db.execute("INSERT INTO members VALUES (?,?,?,?,'member',1,?)",(identifier,name,email,encoded,self.now()))
            db.execute('UPDATE invitations SET consumed=? WHERE id=?',(self.now(),invite['id']))
            db.commit()
            row=db.execute('SELECT * FROM members WHERE id=?',(identifier,)).fetchone()
        return self.user(row)

    def members(self, actor_id):
        with self.connect() as db:
            actor=db.execute('SELECT role FROM members WHERE id=? AND active=1',(actor_id,)).fetchone()
            if not actor:
                raise LabError('This membership is no longer active.', 401)
            rows=[self.user(row) for row in db.execute('SELECT * FROM members WHERE active=1 ORDER BY created')]
            invites=[]
            if actor['role']=='owner':
                invites=[{'id':r['id'],'email':r['email'],'created_at':stamp(r['created']),'expires_at':stamp(r['expires'])}
                         for r in db.execute('SELECT * FROM invitations WHERE consumed IS NULL AND revoked IS NULL AND expires>? ORDER BY created DESC',(self.now(),))]
        return {'members':rows,'invitations':invites}

    def revoke_invitation(self, actor_id, identifier):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE'); self.owner(db,actor_id)
            db.execute('UPDATE invitations SET revoked=? WHERE id=?',(self.now(),identifier)); db.commit()

    def remove_member(self, actor_id, identifier):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE'); self.owner(db,actor_id)
            row=db.execute('SELECT role FROM members WHERE id=? AND active=1',(identifier,)).fetchone()
            if not row: raise LabError('Member was not found.',404)
            if row['role']=='owner': raise LabError('The lab owner cannot be removed.',409)
            db.execute('UPDATE members SET active=0 WHERE id=?',(identifier,))
            db.execute('DELETE FROM sessions WHERE member_id=?',(identifier,))
            db.execute('UPDATE invitations SET revoked=? WHERE email=(SELECT email FROM members WHERE id=?) AND consumed IS NULL',(self.now(),identifier))
            db.commit()

    def record_activity(self, user, action, path, status=None):
        identifier=secrets.token_hex(16)
        with self.connect() as db:
            db.execute('INSERT INTO activity VALUES (?,?,?,?,?,?,?)',(identifier,user['id'],user['name'],action,path,status,self.now()))
            db.commit()
        return identifier

    def finish_activity(self, identifier, status):
        with self.connect() as db:
            db.execute('UPDATE activity SET status=? WHERE id=?',(status,identifier));db.commit()

    def activity(self):
        with self.connect() as db:
            return [{'id':r['id'],'actor_id':r['actor_id'],'actor_name':r['actor_name'],'action':r['action'],
                     'path':r['path'],'status':r['status'],'created_at':stamp(r['created'])}
                    for r in db.execute('SELECT * FROM activity ORDER BY created DESC LIMIT 50')]

    def reset_password(self, email, password):
        email=email_key(email); encoded=password_hash(password)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT id FROM members WHERE email=? AND active=1',(email,)).fetchone()
            if not row: raise LabError('Active member was not found.',404)
            db.execute('UPDATE members SET password=? WHERE id=?',(encoded,row['id']))
            db.execute('DELETE FROM sessions WHERE member_id=?',(row['id'],)); db.commit()


def main():
    parser=argparse.ArgumentParser(description='Local deployment-owner account recovery; no email delivery.')
    parser.add_argument('command',choices=['reset-password'])
    parser.add_argument('--directory',required=True);parser.add_argument('--email',required=True)
    args=parser.parse_args()
    password=getpass.getpass('New password (12–256 characters): ')
    if password != getpass.getpass('Repeat new password: '):
        raise SystemExit('Passwords did not match; no change made.')
    LabAccounts(args.directory).reset_password(args.email,password)
    print('Password changed. Existing sessions for this member have been signed out.')


if __name__=='__main__':
    main()
