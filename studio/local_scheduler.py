"""One bounded CPU scan child for a single-worker standalone web service.

Use one Uvicorn worker and persistent collection storage. A process lock spans
the child lifetime, so a server restart cannot start a second scan. Durable
receipts let the existing collection reconciliation recover acknowledgements.
Logs stay in the private collection store, never the web static directory.
"""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import uuid

from .collection_scheduler import SubmissionUncertain
from .execution import execution_backend, execution_provenance, require_cpu_execution
from .storage import now, read, write

_ID = re.compile(r"[a-f0-9]{32}\Z")
_JOB = re.compile(r"[0-9]{1,39}\Z")
_TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT"}


def _timeout_seconds():
    try:
        seconds = int(os.environ.get("TISSUE_SCAN_TIMEOUT_SECONDS", "3600"))
    except ValueError as error:
        raise ValueError("TISSUE_SCAN_TIMEOUT_SECONDS must be from 30 to 21600") from error
    if not 30 <= seconds <= 21600:
        raise ValueError("TISSUE_SCAN_TIMEOUT_SECONDS must be from 30 to 21600")
    return seconds


def _identity(pid):
    """Linux process birth identity prevents PID reuse from reviving a job."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": pid, "start_ticks": data[19], "state": data[0],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def _same_process(record):
    live = _identity(record["pid"])
    if live:
        return (live["state"] != "Z" and live["start_ticks"] == record.get("start_ticks") and
                live["boot_id"] == record.get("boot_id"))
    try:
        os.kill(record["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    # Without a birth identity, an extant PID is ambiguous after a restart.
    return None


class LocalScanLauncher:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.repository = Path(__file__).resolve().parents[1]
        if self.root.is_relative_to(self.repository / "studio/static"):
            raise ValueError("The collection store and worker logs must be outside web static files")
        self.jobs = self.root / ".local-jobs"
        self.lock = threading.Lock()

    def available(self):
        from .collection_scan import load_configuration
        try:
            self._require_local()
            configuration, _ = load_configuration(self.root)
            return configuration.get("schema_version") == 2 and configuration.get("backend") == "cpu"
        except (OSError, RuntimeError, ValueError, KeyError):
            return False

    @staticmethod
    def _require_local():
        if execution_backend() != "local":
            raise RuntimeError("The local scan worker requires TISSUE_EXECUTION_BACKEND=local")
        return require_cpu_execution()

    @staticmethod
    def _job_id(collection_id, attempt_id):
        if not _ID.fullmatch(str(collection_id)) or not _ID.fullmatch(str(attempt_id)):
            raise ValueError("Invalid scan submission identity")
        # The collection store accepts decimal job IDs. This is an opaque local
        # identifier, never a fabricated Slurm job or operating-system PID.
        return str(int(attempt_id, 16))

    def __call__(self, collection_id, attempt_id=None):
        attempt_id = attempt_id or uuid.uuid4().hex
        job_id = self._job_id(collection_id, attempt_id)
        threads = self._require_local()
        timeout = _timeout_seconds()
        from .collection_scan import snapshot_config
        configuration = snapshot_config(self.root)
        if configuration.get("schema_version") != 2 or configuration.get("backend") != "cpu":
            raise ValueError("The local worker supports only the frozen CPU scan release")
        configuration_sha = configuration.get("sha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", configuration_sha):
            raise ValueError("The collection scan configuration has no frozen identity")
        self.jobs.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock, (self.jobs / ".worker.lock").open("a") as worker:
            folder = self.jobs / job_id
            if folder.exists():
                receipt = read(folder / "job.json")
                if receipt.get("collection_id") != collection_id or receipt.get("attempt_id") != attempt_id:
                    raise ValueError("Local job receipt belongs to a different scan attempt")
                return job_id
            try:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another collection scan is running; retry when it finishes") from error
            folder.mkdir(mode=0o700)
            write(folder / "job.json", {"schema": "local-cpu-scan-job-v1", "job_id": job_id,
                "collection_id": collection_id, "attempt_id": attempt_id,
                "configuration_sha256": configuration_sha, "created_at": now(),
                "threads": threads, "timeout_seconds": timeout, "execution_backend": "local"})
            write(folder / "status.json", {"state": "PENDING", "updated_at": now()})
            environment = dict(os.environ, TISSUE_LOCAL_JOB_ID=job_id)
            try:
                with (folder / "worker.log").open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "studio.local_scheduler", "worker", "--root", str(self.root),
                         "--job-id", job_id, "--lock-fd", str(worker.fileno())],
                        cwd=self.repository, env=environment, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=(worker.fileno(),),
                        start_new_session=True)
            except OSError as error:
                write(folder / "status.json", {"state": "FAILED", "updated_at": now(),
                    "reason": "Local CPU worker could not start"}, overwrite=True)
                raise ValueError("The local CPU worker could not start; retry this scan") from error
            try:
                identity = _identity(process.pid) or {"pid": process.pid}
                write(folder / "launch.json", {**identity, "created_at": now()})
            except OSError as error:
                raise SubmissionUncertain("Local worker started but its acknowledgement was interrupted; refresh to reconcile") from error
            # Reap children without keeping the web process alive at shutdown.
            threading.Thread(target=process.wait, daemon=True, name="scan-reaper").start()
            # Do not unlock: the inherited open-file description holds the lock
            # until the child exits. Closing this parent's descriptor is safe.
            return job_id

    def submit_attempt(self, collection_id, attempt_id):
        return self(collection_id, attempt_id=attempt_id)

    def recover_attempt(self, collection_id, attempt_id, queued_at):
        job_id = self._job_id(collection_id, attempt_id)
        try:
            receipt = read(self.jobs / job_id / "job.json")
            if receipt.get("collection_id") == collection_id and receipt.get("attempt_id") == attempt_id:
                return {"known": True, "job_id": job_id}
        except (OSError, ValueError):
            pass
        return {"known": False, "job_id": None}

    def status(self, job_id):
        unknown = {"state": "UNKNOWN", "known": False, "terminal": False}
        if not _JOB.fullmatch(str(job_id)):
            return unknown
        folder = self.jobs / str(job_id)
        try:
            receipt, status = read(folder / "job.json"), read(folder / "status.json")
            if receipt["job_id"] != str(job_id):
                return unknown
            state = status["state"]
            if state in _TERMINAL:
                return {"state": state, "known": True, "terminal": True}
            process = status.get("process")
            if process is None and (folder / "launch.json").exists():
                process = read(folder / "launch.json")
            if process is not None:
                alive = _same_process(process)
                if alive is True:
                    return {"state": state, "known": True, "terminal": False}
                if alive is None:
                    return unknown
            else:
                # Submission may have died before spawning. If a worker holds
                # the global lock, do not guess which unacknowledged job it is.
                with (self.jobs / ".worker.lock").open("a") as worker:
                    try:
                        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return unknown
            return {"state": "FAILED", "known": True, "terminal": True}
        except (OSError, ValueError, KeyError):
            return unknown


def run_worker(root, job_id, lock_fd):
    launcher = LocalScanLauncher(root)
    launcher._require_local()
    if not _JOB.fullmatch(str(job_id)):
        raise ValueError("Invalid local job identifier")
    folder = launcher.jobs / str(job_id)
    receipt = read(folder / "job.json")
    if launcher._job_id(receipt["collection_id"], receipt["attempt_id"]) != str(job_id):
        raise ValueError("Worker receipt identity differs")
    expected, inherited = (launcher.jobs / ".worker.lock").stat(), os.fstat(lock_fd)
    if (expected.st_dev, expected.st_ino) != (inherited.st_dev, inherited.st_ino):
        raise ValueError("Worker did not inherit the serialized scan lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = _identity(os.getpid()) or {"pid": os.getpid()}
    write(folder / "status.json", {"state": "RUNNING", "updated_at": now(),
        "process": process, **execution_provenance()}, overwrite=True)
    def timed_out(signum, frame):
        raise TimeoutError("Local scan time limit reached; retry to resume verified feature chunks")
    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(receipt.get("timeout_seconds", 3600))
    try:
        from .collection_scan import run
        run(launcher.root, receipt["collection_id"], receipt["configuration_sha256"])
    except Exception as error:
        write(folder / "status.json", {"state": "TIMEOUT" if isinstance(error, TimeoutError) else "FAILED", "updated_at": now(),
            "process": process, **execution_provenance()}, overwrite=True)
        raise
    else:
        write(folder / "status.json", {"state": "COMPLETED", "updated_at": now(),
            "process": process, **execution_provenance()}, overwrite=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--lock-fd", required=True, type=int)
    args = parser.parse_args()
    run_worker(args.root, args.job_id, args.lock_fd)


if __name__ == "__main__":
    main()
