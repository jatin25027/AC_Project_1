"""
Thread-safe job manager for background experiment execution.
"""
import threading
import uuid
import queue
from typing import Dict, Optional


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"   # pending | running | done | error
        self.progress = 0         # 0-100
        self.log_queue: queue.Queue = queue.Queue()
        self.result = None
        self.error = None


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = Job(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def run_in_background(self, job_id: str, func, *args, **kwargs):
        job = self.get_job(job_id)
        if not job:
            return

        def _run():
            job.status = "running"
            try:
                result = func(job, *args, **kwargs)
                job.result = result
                job.status = "done"
                job.log_queue.put({"type": "done", "message": "✅ Experiment complete!", "progress": 100})
            except Exception as exc:
                import traceback
                job.error = str(exc)
                job.status = "error"
                job.log_queue.put({"type": "error", "message": f"❌ Error: {exc}\n{traceback.format_exc()}", "progress": 0})

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()


# Global singleton
job_manager = JobManager()
