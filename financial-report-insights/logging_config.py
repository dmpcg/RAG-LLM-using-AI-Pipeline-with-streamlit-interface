"""
Structured logging configuration.
Uses JSON format in production, human-readable in development.
Controlled via LOG_LEVEL and LOG_FORMAT environment variables.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

# WS-1 P0-12 (2026-05-07): Extended secrets redaction.
# Each pattern matches a recognised secret format and substitutes a labelled
# REDACTED placeholder so log readers can still tell which type of secret was
# present without leaking the value.
_SECRETS_RE = re.compile(
    r"(password|secret|token|api[_-]?key|credential)[=:]\s*\S+",
    re.IGNORECASE,
)

# Bearer JWT / opaque tokens in Authorization headers.
# Matches:  Authorization: Bearer eyJhbGciOiJ...   (JWT)
#           Authorization: Bearer abc123def...     (opaque)
_BEARER_RE = re.compile(
    r"(authorization\s*[:=]\s*)(?:bearer|jwt)\s+([A-Za-z0-9._\-+/=]{8,})",
    re.IGNORECASE,
)

# AWS access keys (AKIA*, ASIA*, AGPA*, AROA*, AIDA*, ANPA*, ANVA*) followed by
# 16 chars; and AWS secret keys (40-char base64-ish).
_AWS_ACCESS_KEY_RE = re.compile(r"\b((?:AKIA|ASIA|AGPA|AROA|AIDA|ANPA|ANVA)[0-9A-Z]{16})\b")
_AWS_SECRET_KEY_RE = re.compile(
    r"(aws_secret_access_key|aws[_-]?secret)\s*[=:]\s*([A-Za-z0-9/+=]{40})",
    re.IGNORECASE,
)

# Bolt / Neo4j connection URLs with embedded credentials:
# bolt://user:password@host:7687, neo4j://user:pw@host, neo4j+s://user:pw@host
_BOLT_URL_RE = re.compile(
    r"(bolt|neo4j(?:\+s|\+ssc)?)://([^:/@\s]+):([^@\s]+)@",
    re.IGNORECASE,
)

# OpenAI / Anthropic SDK keys.  OpenAI: sk-XXXXXXXX (>=20 chars, includes
# project-scoped sk-proj-..., sk-svcacct-...).  Anthropic: sk-ant-... .
_OPENAI_KEY_RE = re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}\b")
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-(?:api\d+|admin\d+)?-?[A-Za-z0-9_\-]{20,}\b")


def _redact(text: str) -> str:
    """Replace recognised secret values with ***REDACTED*** sentinels.

    Covers (since WS-1 P0-12, 2026-05-07):
      * password / secret / token / api_key / credential = value
      * Authorization: Bearer ...
      * AWS access/secret keys
      * bolt:// / neo4j:// URIs with embedded passwords
      * OpenAI sk-... and Anthropic sk-ant-... API keys
    """
    # Order matters: SDK-specific patterns must run before the generic
    # `password|secret|token|api_key|credential` pattern so that
    # ``OPENAI_API_KEY=sk-...`` is recognised as an OpenAI key (preserves
    # the ``sk-***REDACTED***`` shape) rather than the generic redaction.
    text = _ANTHROPIC_KEY_RE.sub("sk-ant-***REDACTED***", text)
    text = _OPENAI_KEY_RE.sub("sk-***REDACTED***", text)
    text = _BEARER_RE.sub(r"\1Bearer ***REDACTED***", text)
    text = _AWS_ACCESS_KEY_RE.sub("***AWS_ACCESS_KEY_REDACTED***", text)
    text = _AWS_SECRET_KEY_RE.sub(r"\1=***REDACTED***", text)
    text = _BOLT_URL_RE.sub(r"\1://\2:***REDACTED***@", text)
    text = _SECRETS_RE.sub(r"\1=***REDACTED***", text)
    return text


class RedactingTextFormatter(logging.Formatter):
    """Text formatter that applies secret redaction."""

    def format(self, record: logging.LogRecord) -> str:
        result = super().format(record)
        return _redact(result)


class JSONFormatter(logging.Formatter):
    """JSON structured log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": _redact(str(record.exc_info[1])),
            }
        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """Configure logging based on environment variables.

    Environment variables:
        LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
        LOG_FORMAT: json, text (default: text)
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.environ.get("LOG_FORMAT", "text").lower()

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(RedactingTextFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
