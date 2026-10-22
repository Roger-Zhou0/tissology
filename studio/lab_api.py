"""Invitation-only shared lab access, wrapping every route and mounted application."""
from __future__ import annotations

import hashlib
import hmac
import logging
import posixpath
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from .lab_accounts import LabAccounts, LabError, SESSION_SECONDS

LOG = logging.getLogger(__name__)
SAFE = {'GET','HEAD','OPTIONS'}
CREDENTIALS = {'/api/lab/setup','/api/lab/login','/api/lab/join'}
PUBLIC = {'/lab','/api/lab/session', '/static/lab.js','/static/lab.css',
          '/static/style.css','/static/presentation.css','/static/navigation.js','/static/lab-session.js'} | CREDENTIALS


class Login(BaseModel):
    model_config = ConfigDict(extra='forbid')
    email: str = Field(max_length=254)
    password: str = Field(min_length=12,max_length=256)


class Setup(Login):
    code: str = Field(min_length=1,max_length=200)
    name: str = Field(min_length=1,max_length=120)
    workspace_name: str = Field(default='Shared lab',min_length=1,max_length=120)


class Join(Login):
    token: str = Field(min_length=1,max_length=200)
    name: str = Field(min_length=1,max_length=120)


class Invitation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    email: str = Field(max_length=254)


def action_name(path, method):
    if path.startswith('/api/lab/invitations'):
        return 'Created an invitation' if method=='POST' else 'Revoked an invitation'
    if path.startswith('/api/lab/members'):
        return 'Removed a lab member'
    if path.endswith('/logout'):
        return 'Signed out'
    if '/teaching/' in path:
        return 'Updated collection teaching'
    if path.endswith('/review') or '/reviews' in path or path.endswith('/check'):
        return 'Saved a review'
    if '/descriptions' in path:
        return 'Saved a finding'
    if '/concepts' in path:
        return 'Saved a search'
    if '/audits' in path:
        return 'Updated an audit'
    if '/studies' in path:
        return 'Updated a study'
    if '/collections' in path:
        return 'Updated a collection'
    return 'Updated shared work'


class LabGate:
    def __init__(self, app, store, cookie_name, public_origin, public_path):
        self.app,self.store,self.cookie_name=app,store,cookie_name
        self.public_origin,self.public_path=public_origin,public_path

    async def __call__(self, scope, receive, send):
        if scope['type']!='http':
            if scope['type']=='websocket':
                await send({'type':'websocket.close','code':1008})
                return
            return await self.app(scope,receive,send)
        request=Request(scope,receive=receive)
        path=scope['path']
        root=scope.get('root_path','').rstrip('/')
        if root and path.startswith(root+'/'):
            path=path[len(root):]
        method=request.method
        session=await run_in_threadpool(self.store.session,request.cookies.get(self.cookie_name))
        scope.setdefault('state',{})['lab_session']=session
        async def safe_send(message):
            if message['type']=='http.response.start':
                headers=MutableHeaders(scope=message)
                headers['Cache-Control']='private, no-store'
                headers['Referrer-Policy']='no-referrer'
                headers['X-Content-Type-Options']='nosniff'
                headers['X-Frame-Options']='SAMEORIGIN'
            await send(message)
        async def reject(status,detail):
            await JSONResponse({'detail':detail},status_code=status)(scope,receive,safe_send)
        if path not in PUBLIC and not session:
            if method in {'GET','HEAD'} and 'text/html' in request.headers.get('accept','') and not path.startswith(('/api/','/static/')):
                # OOD prepends its prefix to root-relative Location headers.
                # A relative redirect survives both direct and portal requests.
                relative=posixpath.relpath('lab',posixpath.dirname(path.lstrip('/')) or '.')
                next_path=path.lstrip('/') or 'workspace'
                if scope.get('query_string'):
                    next_path+='?'+scope['query_string'].decode('latin1')
                target=relative+'?'+urlencode({'view':'login','next':next_path})
                await RedirectResponse(target,status_code=303)(scope,receive,safe_send)
            else:
                await reject(401,'Sign in to the shared lab to continue.')
            return
        if method not in SAFE:
            origin=request.headers.get('origin')
            if (origin and origin!=self.public_origin) or request.headers.get('sec-fetch-site')=='cross-site':
                await reject(403,'This request must come from the lab page.');return
            if path in CREDENTIALS:
                if request.headers.get('x-lab-request')!='1' or request.headers.get('content-type','').split(';')[0].strip()!='application/json':
                    await reject(403,'Open the lab sign-in page and try again.');return
                # Read a small, bounded credential body before JSON/Pydantic parse.
                body=b''
                while True:
                    message=await receive()
                    if message['type']=='http.disconnect':return
                    body+=message.get('body',b'')
                    if len(body)>8192:
                        await reject(413,'The sign-in request is too large.');return
                    if not message.get('more_body',False):break
                delivered=False
                original_receive=receive
                async def replay():
                    nonlocal delivered
                    if delivered:return await original_receive()
                    delivered=True
                    return {'type':'http.request','body':body,'more_body':False}
                receive=replay
            elif not session or not hmac.compare_digest(request.headers.get('x-csrf-token','').encode('utf-8'),session['csrf_token'].encode('utf-8')):
                await reject(403,'Your session changed. Reload this page before saving.');return
        activity_id=None
        if method not in SAFE and session and path not in CREDENTIALS:
            # Reserve attribution before any scientific write. Never log the body,
            # query string, credentials, cookies or invitation token.
            try:
                activity_id=await run_in_threadpool(self.store.record_activity,session['user'],action_name(path,method),path[:500])
            except Exception:
                LOG.exception('Could not reserve lab activity record; request was not performed')
                await reject(503,'Activity could not be recorded. Try again shortly.');return
        async def tracked_send(message):
            if activity_id and message['type']=='http.response.start':
                try:await run_in_threadpool(self.store.finish_activity,activity_id,message['status'])
                except Exception:
                    LOG.exception('Lab activity completion failed; original application response retained')
            await safe_send(message)
        await self.app(scope,receive,tracked_send)


def register_lab(app, directory, *, public_origin, public_path='/', secure_cookie=True):
    origin=urlsplit(public_origin)
    if origin.scheme not in {'http','https'} or not origin.netloc or origin.path not in ('','/') or origin.query or origin.fragment or origin.username:
        raise ValueError('Set an explicit public lab origin, without a path')
    public_origin=public_origin.rstrip('/')
    if secure_cookie and origin.scheme!='https':
        raise ValueError('Secure lab sessions require an HTTPS public origin')
    if not public_path.startswith('/') or any(x in public_path for x in ('..','?','#','\\','%','//')):
        raise ValueError('Invalid public lab path')
    public_path=public_path.rstrip('/')+'/'
    store=LabAccounts(directory)
    cookie='tissue_lab_'+hashlib.sha256((str(store.directory)+public_path).encode()).hexdigest()[:12]
    app.state.lab=store;app.state.lab_cookie_name=cookie
    app.add_middleware(LabGate,store=store,cookie_name=cookie,public_origin=public_origin,public_path=public_path)

    @app.exception_handler(LabError)
    async def lab_error(request,error):
        return JSONResponse({'detail':str(error)},status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,error):
        # Pydantic's default response includes the submitted input. Credentials
        # must never be reflected into browser errors, screenshots or telemetry.
        path=request.scope['path']
        root=request.scope.get('root_path','').rstrip('/')
        if root and path.startswith(root+'/'):
            path=path[len(root):]
        if path.startswith('/api/lab/'):
            return JSONResponse({'detail':'Check the account fields and try again.'},status_code=422)
        return await request_validation_exception_handler(request,error)

    def session_view(current):
        return {'configured':True,**store.info(),'authenticated':bool(current),'public_path':public_path,
                'user':current['user'] if current else None,**({'csrf_token':current['csrf_token']} if current else {})}

    def signed_in(request):
        current=request.state.lab_session
        if not current:raise LabError('Sign in to continue.',401)
        return current['user']

    def authenticated_response(user,action):
        token,csrf=store.create_session(user['id'])
        try:
            store.record_activity(user,action,'/lab',200)
        except Exception:
            # Account creation/join has already committed. An activity failure
            # must not turn success into a misleading invitation/setup retry.
            LOG.exception('Lab sign-in activity could not be recorded; account response retained')
        response=JSONResponse(session_view({'user':user,'csrf_token':csrf}))
        response.set_cookie(cookie,token,max_age=SESSION_SECONDS,httponly=True,secure=secure_cookie,samesite='strict',path=public_path)
        return response

    @app.get('/lab')
    def lab_page():
        return FileResponse(Path(__file__).parent/'static/lab.html')

    @app.get('/api/lab/session')
    def session(request:Request):
        return session_view(request.state.lab_session)

    @app.post('/api/lab/setup')
    def setup(payload:Setup):
        return authenticated_response(store.setup(**payload.model_dump()),'Created the shared lab')

    @app.post('/api/lab/login')
    def login(payload:Login):
        return authenticated_response(store.login(**payload.model_dump()),'Signed in')

    @app.post('/api/lab/join')
    def join(payload:Join):
        return authenticated_response(store.join(**payload.model_dump()),'Joined the lab')

    @app.post('/api/lab/logout')
    def logout(request:Request):
        store.logout(request.cookies.get(cookie,''))
        response=JSONResponse({'ok':True})
        response.delete_cookie(cookie,path=public_path,secure=secure_cookie,httponly=True,samesite='strict')
        return response

    @app.get('/api/lab/members')
    def members(request:Request):
        return store.members(signed_in(request)['id'])

    @app.post('/api/lab/invitations')
    def invitation(payload:Invitation,request:Request):
        return store.create_invitation(signed_in(request)['id'],payload.email)

    @app.delete('/api/lab/invitations/{identifier}')
    def revoke(identifier:str,request:Request):
        store.revoke_invitation(signed_in(request)['id'],identifier);return {'ok':True}

    @app.delete('/api/lab/members/{identifier}')
    def remove(identifier:str,request:Request):
        store.remove_member(signed_in(request)['id'],identifier);return {'ok':True}

    @app.get('/api/lab/activity')
    def activity(request:Request):
        signed_in(request)
        return {'items':store.activity()}

    return store
