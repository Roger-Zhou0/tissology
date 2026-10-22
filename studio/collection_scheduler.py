"""Submit fixed, serialized collection scans to Delta and expose job state.

The HTTP process never loads the image model. Paths and the worker script are
server configuration; a request contributes only a validated collection ID.
"""
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import subprocess
import threading
import time


class SubmissionUncertain(ValueError):
    """Slurm may have accepted a job despite an interrupted acknowledgement."""


class SlurmScanLauncher:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.repository = Path(__file__).resolve().parents[1]
        self.script = self.repository / 'scripts/collection_scan.sbatch'
        self.cpu_script = self.repository / 'scripts/collection_scan_cpu.sbatch'
        self.cache = {}
        self.lock = threading.Lock()
        self.attempt_cache = {}

    def available(self):
        """Check configured backend without queuing work or creating a snapshot."""
        from .collection_scan import load_configuration
        try:
            configuration, _ = load_configuration(self.root)
            if configuration.get('schema_version') == 2 and configuration.get('backend') == 'cpu':
                return self.cpu_script.is_file()
            if configuration.get('schema_version') == 1 and configuration.get('backend', 'cuda') == 'cuda':
                return self.script.is_file()
            return False
        except (OSError, ValueError, KeyError):
            return False

    def __call__(self, collection_id, attempt_id=None):
        if not re.fullmatch(r'[a-f0-9]{32}', str(collection_id)):
            raise ValueError('Invalid collection identifier')
        if not (self.root / 'scan-model.json').is_file():
            raise ValueError('The collection scan model has not been configured')
        comment = self.attempt_comment(collection_id, attempt_id) if attempt_id is not None else None
        from .collection_scan import snapshot_config
        configuration = snapshot_config(self.root)
        configuration_sha = configuration.get('sha256', '')
        if not re.fullmatch('[a-f0-9]{64}', configuration_sha):
            raise ValueError('The collection scan configuration has no frozen identity')
        version, backend = configuration.get('schema_version'), configuration.get('backend', 'cuda')
        if version == 2 and backend == 'cpu':
            script = self.cpu_script
            resources = ['--account=bfqi-delta-cpu', '--partition=cpu-interactive',
                         '--cpus-per-task=8', '--mem=16G', '--time=00:45:00']
        elif version == 1 and backend == 'cuda':
            script = self.script
            resources = ['--account=bfqi-delta-gpu']
        else:
            raise ValueError('Unsupported collection scan execution backend')
        if not script.is_file():
            raise ValueError('The scheduled scan worker is unavailable')
        arguments = ['sbatch', '--parsable', '--dependency=singleton', '--job-name=tissue-collection-scan']
        arguments.extend(resources)
        if comment is not None:
            arguments.append('--comment=' + comment)
        # Server-side SBATCH_* defaults must not introduce another account or GPU
        # request. All scheduling options come from these fixed arguments/scripts.
        environment = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
        if backend == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = ''
        try:
            result = subprocess.run(
                arguments + [str(script), str(self.root), collection_id, configuration_sha],
                cwd=self.repository, capture_output=True, text=True,
                timeout=20, check=False, env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise SubmissionUncertain('Scheduler acknowledgement timed out; the submission must be reconciled before retrying.') from error
        except OSError as error:
            raise ValueError('The compute scheduler could not accept this scan. Retry when it is available.') from error
        identifier = result.stdout.strip().split(';', 1)[0]
        if result.returncode or not identifier.isdigit():
            # Keep scheduler/environment diagnostics in server logs, not image
            # metadata or the browser. The caller records a recoverable failure.
            raise SubmissionUncertain('The compute scheduler rejected or did not confirm this scan; check the recorded attempt before retrying.')
        return identifier

    @staticmethod
    def attempt_comment(collection_id, attempt_id):
        if not re.fullmatch('[a-f0-9]{32}', str(collection_id)) or not re.fullmatch('[a-f0-9]{32}', str(attempt_id)):
            raise ValueError('Invalid scan submission identity')
        return 'tissue-studio:' + collection_id + ':' + attempt_id

    def submit_attempt(self, collection_id, attempt_id):
        return self(collection_id, attempt_id=attempt_id)

    def recover_attempt(self, collection_id, attempt_id, queued_at):
        """Exact comments are stored by Delta (AccountingStoreFlags=job_comment)."""
        comment = self.attempt_comment(collection_id, attempt_id)
        cached = self.attempt_cache.get(comment)
        if cached and time.monotonic() - cached[0] < 15:
            return dict(cached[1])
        unresolved = {'known': False, 'job_id': None}
        try:
            start = (datetime.fromisoformat(queued_at) - timedelta(days=1)).strftime('%Y-%m-%d')
            active = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i|%k'],
                capture_output=True, text=True, timeout=5, check=False)
            ids = {parts[0] for line in active.stdout.splitlines()
                   if len(parts := line.split('|')) >= 2 and parts[0].isdigit() and parts[1].strip() == comment}
            if not ids and active.returncode == 0:
                history = subprocess.run(['sacct', '--noheader', '--parsable2', '--allocations',
                    '--user', str(os.getuid()), '--starttime', start, '--format=JobIDRaw,Comment%200'],
                    capture_output=True, text=True, timeout=5, check=False)
                if history.returncode == 0:
                    ids = {parts[0] for line in history.stdout.splitlines()
                           if len(parts := line.split('|')) >= 2 and parts[0].isdigit() and parts[1].strip() == comment}
            result = {'known': True, 'job_id': next(iter(ids))} if len(ids) == 1 else unresolved
        except (OSError, subprocess.TimeoutExpired, ValueError):
            result = unresolved
        self.attempt_cache[comment] = (time.monotonic(), result)
        return dict(result)

    def status(self, job_id):
        identifier = str(job_id)
        if not identifier.isdigit():
            return {'state': 'UNKNOWN', 'known': False, 'terminal': False}
        with self.lock:
            cached = self.cache.get(identifier)
            if cached and time.monotonic() - cached[0] < 15:
                return dict(cached[1])
            result = self._status(identifier)
            if len(self.cache) > 128:
                self.cache.clear()
            self.cache[identifier] = (time.monotonic(), result)
            return dict(result)

    @staticmethod
    def _status(identifier):
        unknown = {'state': 'UNKNOWN', 'known': False, 'terminal': False}
        try:
            active = subprocess.run(
                ['squeue', '--noheader', '--jobs', identifier, '--format=%T'],
                capture_output=True, text=True, timeout=5, check=False,
            )
            state = active.stdout.strip().splitlines()[0] if active.returncode == 0 and active.stdout.strip() else ''
            if not state:
                history = subprocess.run(
                    ['sacct', '--noheader', '--parsable2', '--jobs', identifier,
                     '--format=JobIDRaw,State'],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                if history.returncode:
                    return unknown
                for line in history.stdout.splitlines():
                    fields = line.split('|')
                    if len(fields) >= 2 and fields[0] == identifier:
                        state = fields[1]
                        break
            if not state:
                return unknown
            state = state.split()[0].rstrip('+')
            terminal = state in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT',
                                'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL',
                                'DEADLINE', 'PREEMPTED', 'REVOKED'}
            return {'state': state, 'known': True, 'terminal': terminal}
        except (OSError, subprocess.TimeoutExpired):
            return unknown
