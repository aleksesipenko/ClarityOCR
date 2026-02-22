import ipaddress
import os
import socket
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import urlparse


class SecurityValidationError(ValueError):
    """Raised when input fails v2 security checks."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _is_blocked_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_host_ips(hostname: str) -> set[str]:
    infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    ips: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        ip_text = sockaddr[0]
        ips.add(ip_text)
    return ips


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SecurityValidationError("Only http/https URLs are supported")
    if not parsed.hostname:
        raise SecurityValidationError("URL must include a hostname")

    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        raise SecurityValidationError("URL host is not allowed")

    try:
        ips = _resolve_host_ips(host)
    except socket.gaierror as exc:
        raise SecurityValidationError(f"Cannot resolve host: {host}") from exc

    if not ips:
        raise SecurityValidationError(f"Host has no resolvable IPs: {host}")

    for ip_text in ips:
        if _is_blocked_ip(ip_text):
            raise SecurityValidationError(f"URL host resolves to blocked IP: {ip_text}")


def _validate_zip_safety(path: Path) -> None:
    max_depth = _env_int("V2_MAX_ZIP_DEPTH", 6)
    max_entries = _env_int("V2_MAX_ZIP_ENTRIES", 10000)
    max_total_mb = _env_int("V2_MAX_ZIP_UNCOMPRESSED_MB", 1024)
    max_ratio = _env_int("V2_MAX_ZIP_RATIO", 200)

    max_total_bytes = max_total_mb * 1024 * 1024

    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > max_entries:
                raise SecurityValidationError(f"ZIP has too many entries: {len(infos)}")

            total_uncompressed = 0
            for info in infos:
                normalized = info.filename.replace("\\", "/").strip("/")
                if not normalized:
                    continue

                parts = PurePosixPath(normalized).parts
                if ".." in parts:
                    raise SecurityValidationError("ZIP contains path traversal entries")

                depth = normalized.count("/")
                if depth > max_depth:
                    raise SecurityValidationError(f"ZIP nesting depth exceeds limit: {depth}")

                if info.file_size < 0 or info.compress_size < 0:
                    raise SecurityValidationError("ZIP entry has invalid size fields")

                total_uncompressed += info.file_size
                if total_uncompressed > max_total_bytes:
                    raise SecurityValidationError("ZIP uncompressed size exceeds limit")

                ratio = info.file_size / max(1, info.compress_size)
                if ratio > max_ratio:
                    raise SecurityValidationError("ZIP compression ratio exceeds safety limit")
    except zipfile.BadZipFile as exc:
        raise SecurityValidationError("Invalid ZIP archive") from exc


def _validate_local_path(path_text: str) -> None:
    path = Path(path_text)
    if not path.exists():
        return

    max_file_size_mb = _env_int("V2_MAX_INPUT_FILE_SIZE_MB", 512)
    max_file_bytes = max_file_size_mb * 1024 * 1024

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SecurityValidationError(f"Cannot stat file: {path_text}") from exc

    if size > max_file_bytes:
        raise SecurityValidationError(f"Input file exceeds size limit: {path_text}")

    if path.suffix.lower() == ".zip":
        _validate_zip_safety(path)


def validate_inputs_security(inputs: Iterable[str]) -> None:
    for item in inputs:
        if _is_url(item):
            validate_public_url(item)
        else:
            _validate_local_path(item)


def validate_callback_url(callback_url: str) -> None:
    validate_public_url(callback_url)
