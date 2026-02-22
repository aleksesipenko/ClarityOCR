import os
import json
from datetime import datetime, timezone

def py_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Boolean, 
    Text, ForeignKey, UniqueConstraint, event, Engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

# =============================================================================
# MODELS
# =============================================================================

class Job(Base):
    __tablename__ = 'jobs'
    
    job_id = Column(String, primary_key=True)
    client_id = Column(String, nullable=False, default='default')
    client_request_id = Column(String, nullable=False)
    payload_hash = Column(String, nullable=False) # sha256 of normalized request body
    mode = Column(String, nullable=False)
    preset = Column(String, nullable=False)
    status = Column(String, nullable=False, index=True) # queued, running, completed, partial, failed, canceled
    created_at = Column(DateTime, nullable=False, default=py_utc_now)
    accepted_at = Column(DateTime, nullable=True)
    config = Column(Text, nullable=True) # JSON config
    version = Column(Integer, nullable=False, default=1) # Optimistic locking
    
    __table_args__ = (
        UniqueConstraint('client_id', 'client_request_id', name='uq_job_client_request'),
    )

    files = relationship("JobFile", back_populates="job", cascade="all, delete-orphan")
    events = relationship("JobEvent", back_populates="job", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="job", cascade="all, delete-orphan")


class JobFile(Base):
    __tablename__ = 'job_files'
    
    id = Column(String, primary_key=True) # uuid
    job_id = Column(String, ForeignKey('jobs.job_id'), nullable=False, index=True)
    input_path = Column(String, nullable=False)
    status = Column(String, nullable=False, index=True) # queued, running, completed, failed_recoverable, failed_final, canceled
    attempt = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    last_error_code = Column(String, nullable=True)
    last_error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    version = Column(Integer, nullable=False, default=1) # Optimistic locking
    
    job = relationship("Job", back_populates="files")
    artifacts = relationship("Artifact", back_populates="file", cascade="all, delete-orphan")
    events = relationship("JobEvent", back_populates="file", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = 'artifacts'
    
    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey('jobs.job_id'), nullable=True, index=True)
    file_id = Column(String, ForeignKey('job_files.id'), nullable=True, index=True)
    type = Column(String, nullable=False) # md, meta, naming, pdf, report, raw.md
    path = Column(String, nullable=False)
    sha256 = Column(String, nullable=True)
    
    job = relationship("Job", back_populates="artifacts")
    file = relationship("JobFile", back_populates="artifacts")


class JobEvent(Base):
    __tablename__ = 'job_events'
    
    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey('jobs.job_id'), nullable=False, index=True)
    file_id = Column(String, ForeignKey('job_files.id'), nullable=True, index=True)
    event_code = Column(String, nullable=False) # e.g. queued, running, file_done, failed, worker_crash
    timestamp = Column(DateTime, nullable=False, default=py_utc_now)
    correlation_id = Column(String, nullable=True)
    payload = Column(Text, nullable=True) # JSON details
    
    job = relationship("Job", back_populates="events")
    file = relationship("JobFile", back_populates="events")


class Worker(Base):
    __tablename__ = 'workers'
    
    worker_id = Column(String, primary_key=True)
    gpu_id = Column(String, nullable=True)
    last_heartbeat = Column(DateTime, nullable=False, default=py_utc_now)


# =============================================================================
# DATABASE SETUP
# =============================================================================

DB_BUSY_TIMEOUT_MS = 5000

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    cursor.close()

def get_engine(db_path: str = None):
    """Get SQLAlchemy engine configured for SQLite with WAL."""
    if not db_path:
        # Default to a db file in the user's home or project root depending on environment
        # For simplicity, put it in the same dir as the module or user data dir.
        data_dir = Path.home() / ".clarityocr"
        os.makedirs(data_dir, exist_ok=True)
        db_path = str(data_dir / "clarity_v2.db")
    
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={'check_same_thread': False}, # Needed for FastAPI depending on setup
        echo=False
    )
    return engine

def init_db(db_path: str = None):
    """Initialize the database schema."""
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    return engine

def get_session_maker(engine):
    """Get a configured session maker."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global instances for the application
db_engine = None
SessionLocal = None

def setup_db(db_path: str = None):
    global db_engine, SessionLocal
    resolved_db_path = db_path or os.getenv("DB_PATH")
    db_engine = init_db(resolved_db_path)
    SessionLocal = get_session_maker(db_engine)

def get_session():
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call setup_db() first.")
    return SessionLocal()


def close_db():
    global db_engine, SessionLocal
    if db_engine is not None:
        db_engine.dispose()
    db_engine = None
    SessionLocal = None
