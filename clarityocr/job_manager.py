import uuid
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from .db import Job, JobFile, Artifact, JobEvent, Worker

class JobConflictError(Exception):
    """Raised when an optimistic lock conflict occurs."""
    pass

class IdempotencyError(Exception):
    """Raised when the client_id/client_request_id matches but payload_hash differs."""
    pass


def normalize_payload_for_hash(payload: Dict[str, Any]) -> str:
    """Normalize a request payload and return its SHA256 hash."""
    # Sort keys to ensure deterministic serialization
    payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()

def create_job(
    session: Session, 
    client_id: str, 
    client_request_id: str, 
    payload: Dict[str, Any], 
    files: List[str], 
    mode: str, 
    preset: str
) -> Job:
    """Create a new job or return an existing one if idempotency key matches."""
    payload_hash = normalize_payload_for_hash(payload)
    
    # Check for existing job
    existing_job = session.query(Job).filter_by(
        client_id=client_id, 
        client_request_id=client_request_id
    ).first()
    
    if existing_job:
        if existing_job.payload_hash != payload_hash:
            raise IdempotencyError(f"Idempotency conflict for request {client_request_id}")
        return existing_job
        
    job_id = str(uuid.uuid4())
    
    job = Job(
        job_id=job_id,
        client_id=client_id,
        client_request_id=client_request_id,
        payload_hash=payload_hash,
        mode=mode,
        preset=preset,
        status="queued",
        config=json.dumps(payload)
    )
    session.add(job)
    
    # Create file records
    if "merge" in mode:
        job_file = JobFile(
            id=str(uuid.uuid4()),
            job_id=job_id,
            input_path="<merge_job>",
            status="queued"
        )
        session.add(job_file)
    else:
        for f in files:
            job_file = JobFile(
                id=str(uuid.uuid4()),
                job_id=job_id,
                input_path=f,
                status="queued"
            )
            session.add(job_file)
        
    # Append event
    event = JobEvent(
        id=str(uuid.uuid4()),
        job_id=job_id,
        event_code="job_queued",
        payload=json.dumps({"file_count": len(files)})
    )
    session.add(event)
    
    session.commit()
    session.refresh(job)
    return job

def append_event(session: Session, job_id: str, event_code: str, file_id: Optional[str] = None, correlation_id: Optional[str] = None, payload: Optional[Dict] = None):
    """Append a structured event to the job log."""
    event = JobEvent(
        id=str(uuid.uuid4()),
        job_id=job_id,
        file_id=file_id,
        event_code=event_code,
        correlation_id=correlation_id,
        payload=json.dumps(payload) if payload else None
    )
    session.add(event)
    session.commit()

def transition_file_status(session: Session, file_id: str, new_status: str, max_retries: int = 3, **kwargs) -> JobFile:
    """
    Update a file's status with optimistic locking.
    Retries up to max_retries if StaleDataError occurs.
    """
    for attempt in range(max_retries):
        try:
            db_file = session.query(JobFile).filter_by(id=file_id).one()
            old_version = db_file.version
            
            # Application-level valid transitions
            # queued -> running | canceled
            # running -> completed | failed_recoverable | failed_final | canceled | queued
            # failed_recoverable -> queued | canceled
            # failed_final | canceled -> queued (manual retry)
            valid_transition = False
            if db_file.status == "queued" and new_status in ["running", "canceled"]:
                valid_transition = True
            elif db_file.status == "running" and new_status in ["completed", "failed_recoverable", "failed_final", "canceled", "queued"]:
                valid_transition = True
            elif db_file.status == "failed_recoverable" and new_status == "canceled":
                valid_transition = True
            elif db_file.status in ["failed_recoverable", "failed_final", "canceled"] and new_status == "queued":
                valid_transition = True
                db_file.attempt += 1
            
            if not valid_transition:
                raise ValueError(f"Invalid file transition {db_file.status} -> {new_status}")
                
            db_file.status = new_status
            db_file.version = old_version + 1
            
            # Apply any extra fields
            for k, v in kwargs.items():
                setattr(db_file, k, v)
                
            session.commit()
            return db_file
            
        except StaleDataError:
            session.rollback()
            if attempt == max_retries - 1:
                raise JobConflictError(f"Optimistic lock failed for file {file_id}")
            continue

def transition_job_status(session: Session, job_id: str, new_status: str, max_retries: int = 3) -> Job:
    """Update a job's status with optimistic locking."""
    for attempt in range(max_retries):
        try:
            job = session.query(Job).filter_by(job_id=job_id).one()
            old_version = job.version
            
            valid_transition = False
            if job.status == "queued" and new_status in ["running", "canceled"]:
                valid_transition = True
            elif job.status == "running" and new_status in ["completed", "partial", "failed", "canceled"]:
                valid_transition = True
            elif job.status in ["canceled", "failed", "partial"] and new_status == "running":
                valid_transition = True  # Retry failed/canceled
            elif job.status in ["failed", "partial"] and new_status == "canceled":
                valid_transition = True  # Allow idempotent user cancellation races
                
            if not valid_transition:
                raise ValueError(f"Invalid job transition {job.status} -> {new_status}")
                
            job.status = new_status
            job.version = old_version + 1
            
            if new_status == "running" and job.accepted_at is None:
                job.accepted_at = datetime.now(timezone.utc).replace(tzinfo=None)
                
            session.commit()
            return job
            
        except StaleDataError:
            session.rollback()
            if attempt == max_retries - 1:
                raise JobConflictError(f"Optimistic lock failed for job {job_id}")
            continue

def check_and_update_job_completion(session: Session, job_id: str):
    """
    Calculate the job's final status based on file statuses,
    and transition the job if all files are resolved.
    """
    files = session.query(JobFile).filter_by(job_id=job_id).all()
    if not files:
        return
        
    resolved_count = 0
    success_count = 0
    failed_count = 0
    canceled_count = 0
    
    for f in files:
        if f.status in ["completed", "failed_final", "canceled"]:
            resolved_count += 1
            if f.status == "completed":
                success_count += 1
            elif f.status == "failed_final":
                failed_count += 1
            elif f.status == "canceled":
                canceled_count += 1
                
    if resolved_count < len(files):
        # Job still running
        return
        
    # Determine final status
    final_status = "completed"
    if canceled_count == len(files):
        final_status = "canceled"
    elif failed_count == len(files):
        final_status = "failed"
    elif failed_count > 0:
        final_status = "partial"
        
    try:
        transition_job_status(session, job_id, final_status)
        append_event(session, job_id, f"job_{final_status}")
    except ValueError:
        pass # Already in a final state or invalid transition


def update_worker_heartbeat(session: Session, worker_id: str, gpu_id: Optional[str] = None):
    """Update or create worker heartbeat."""
    worker = session.query(Worker).filter_by(worker_id=worker_id).first()
    if not worker:
        worker = Worker(worker_id=worker_id, gpu_id=gpu_id, last_heartbeat=datetime.now(timezone.utc).replace(tzinfo=None))
        session.add(worker)
    else:
        worker.last_heartbeat = datetime.now(timezone.utc).replace(tzinfo=None)
        if gpu_id is not None:
            worker.gpu_id = gpu_id
    session.commit()

def record_artifact(session: Session, job_id: str, file_id: str, type: str, path: str, sha256: str = None):
    """Save an artifact record to the db."""
    artifact = Artifact(
        id=str(uuid.uuid4()),
        job_id=job_id,
        file_id=file_id,
        type=type,
        path=path,
        sha256=sha256
    )
    session.add(artifact)
    session.commit()
    return artifact
