from __future__ import annotations

import os
import smtplib
import socket
import threading
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import tomllib


_THREAD_HOOK_INSTALLED = False


@dataclass(frozen=True)
class ErrorNotifierConfig:
    enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    mail_to: str
    mail_from: str
    min_http_status: int


def _read_secrets_file() -> dict[str, Any]:
    root_dir = Path(__file__).resolve().parents[3]
    secrets_path = root_dir / "secrets.toml"
    if not secrets_path.exists():
        return {}
    with secrets_path.open("rb") as f:
        return tomllib.load(f)


def _get_secret(name: str, secrets: dict[str, Any], default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and str(value).strip():
        return str(value).strip()
    value = secrets.get(name)
    if value is not None and str(value).strip():
        return str(value).strip()
    return default


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def get_error_notifier_config() -> ErrorNotifierConfig:
    secrets = _read_secrets_file()
    enabled = _truthy(_get_secret("ERROR_NOTIFY_ENABLED", secrets), default=False)
    smtp_host = _get_secret("ERROR_NOTIFY_SMTP_HOST", secrets, "smtp.gmail.com") or "smtp.gmail.com"
    smtp_port_raw = _get_secret("ERROR_NOTIFY_SMTP_PORT", secrets, "587") or "587"
    smtp_username = _get_secret("ERROR_NOTIFY_SMTP_USERNAME", secrets, "") or ""
    smtp_password = _get_secret("ERROR_NOTIFY_SMTP_APP_PASSWORD", secrets, "") or ""
    mail_to = _get_secret("ERROR_NOTIFY_TO", secrets, "") or ""
    mail_from = _get_secret("ERROR_NOTIFY_FROM", secrets, smtp_username) or smtp_username
    min_status_raw = _get_secret("ERROR_NOTIFY_MIN_HTTP_STATUS", secrets, "500") or "500"

    try:
        smtp_port = int(smtp_port_raw)
    except Exception:
        smtp_port = 587

    try:
        min_http_status = int(min_status_raw)
    except Exception:
        min_http_status = 500

    return ErrorNotifierConfig(
        enabled=enabled,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        mail_to=mail_to,
        mail_from=mail_from,
        min_http_status=min_http_status,
    )


def is_error_notifier_configured() -> bool:
    cfg = get_error_notifier_config()
    return bool(
        cfg.enabled
        and cfg.smtp_host
        and cfg.smtp_port
        and cfg.smtp_username
        and cfg.smtp_password
        and cfg.mail_to
        and cfg.mail_from
    )


def _build_message(
    subject: str,
    body: str,
    cfg: ErrorNotifierConfig,
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject[:180]
    msg["From"] = cfg.mail_from
    msg["To"] = cfg.mail_to
    msg.set_content(body)
    return msg


def _send_email(subject: str, body: str) -> None:
    cfg = get_error_notifier_config()
    if not is_error_notifier_configured():
        return

    msg = _build_message(subject, body, cfg)
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as smtp:
        smtp.starttls()
        smtp.login(cfg.smtp_username, cfg.smtp_password)
        smtp.send_message(msg)


def _send_email_best_effort(subject: str, body: str) -> None:
    try:
        _send_email(subject, body)
    except Exception:
        # Never let notification failures become application failures.
        pass


def send_error_email_async(subject: str, body: str) -> None:
    if not is_error_notifier_configured():
        return
    threading.Thread(
        target=_send_email_best_effort,
        args=(subject, body),
        daemon=True,
    ).start()


def format_exception_report(
    *,
    title: str,
    exc: BaseException,
    request: Any | None = None,
    status_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[str, str]:
    now = datetime.now(UTC).isoformat()
    host = socket.gethostname()
    exc_name = exc.__class__.__name__
    subject_status = f" HTTP {status_code}" if status_code is not None else ""
    subject = f"[AryaPhoneSupp]{subject_status} {exc_name}: {str(exc)[:80]}"

    lines = [
        title,
        "",
        f"time_utc: {now}",
        f"host: {host}",
        f"exception_type: {exc_name}",
        f"exception_message: {exc}",
    ]
    if status_code is not None:
        lines.append(f"http_status: {status_code}")

    if request is not None:
        lines.extend([
            "",
            "request:",
            f"  method: {getattr(request, 'method', '-')}",
            f"  url: {getattr(request, 'url', '-')}",
            f"  client: {getattr(getattr(request, 'client', None), 'host', '-')}",
        ])

    if extra:
        lines.append("")
        lines.append("extra:")
        for key, value in extra.items():
            lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "traceback:",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ])
    return subject, "\n".join(lines)


def notify_exception(
    *,
    title: str,
    exc: BaseException,
    request: Any | None = None,
    status_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    cfg = get_error_notifier_config()
    if status_code is not None and status_code < cfg.min_http_status:
        return
    subject, body = format_exception_report(
        title=title,
        exc=exc,
        request=request,
        status_code=status_code,
        extra=extra,
    )
    send_error_email_async(subject, body)


def install_thread_exception_notifier() -> None:
    global _THREAD_HOOK_INSTALLED
    if _THREAD_HOOK_INSTALLED:
        return

    previous_hook = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        exc = args.exc_value
        if exc is not None:
            notify_exception(
                title="Unhandled background thread exception",
                exc=exc,
                extra={"thread_name": getattr(args.thread, "name", "-")},
            )
        previous_hook(args)

    threading.excepthook = _hook
    _THREAD_HOOK_INSTALLED = True
