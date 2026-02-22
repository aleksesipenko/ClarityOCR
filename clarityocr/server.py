#!/usr/bin/env python3
"""
ClarityOCR V2 Web Server.

Server-first entrypoint for VPS/Docker deployments with local debug support.
Only V2 API endpoints are exposed.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from clarityocr.api_v2 import router as api_v2_router
from clarityocr.db import close_db, setup_db
from clarityocr.pipeline_v2 import start_workers, stop_workers


APP_HOST = os.getenv("CLARITY_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("CLARITY_PORT", "8008"))

security = HTTPBasic(auto_error=False)


def get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "http://127.0.0.1:8008",
        "http://localhost:8008",
    ]


def get_worker_count() -> int:
    raw = os.getenv("V2_WORKERS", "1").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1
    return 1 if value < 1 else value


def get_current_user(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> str:
    api_user = os.getenv("API_USER")
    api_pass = os.getenv("API_PASSWORD")

    if not api_user or not api_pass:
        return "anonymous"

    if credentials is None or credentials.username != api_user or credentials.password != api_pass:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_db()
    start_workers(num_workers=get_worker_count())
    try:
        yield
    finally:
        stop_workers()
        close_db()


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "web" / "static"


app = FastAPI(title="ClarityOCR V2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v2_router, prefix="/api/v2", dependencies=[Depends(get_current_user)])

_static_path = static_dir()
app.mount("/static", StaticFiles(directory=str(_static_path)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_static_path / "index.html"))


@app.get("/api/health")
def api_health() -> JSONResponse:
    return JSONResponse({"status": "ok", "api": "v2"})


def run_server(host: str = APP_HOST, port: int = APP_PORT) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run_server(host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()
