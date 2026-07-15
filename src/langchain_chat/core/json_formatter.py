"""Custom JSON formatter for structured logging + API key masking.

Provides:
- :func:`mask_api_key`: redact ``sk-...`` patterns in log messages.
- :class:`JsonFormatter`: emit each log record as a single-line JSON object.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

# sk- followed by 10+ alphanumeric characters
_API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{10,}")


def mask_api_key(text: str) -> str:
    """Replace API keys in *text* with a truncated safe representation.

    ``sk-abcdef1234567890`` → ``sk-abcdef...``
    """
    return _API_KEY_PATTERN.sub(lambda m: m.group()[:8] + "...", text)


class JsonFormatter(logging.Formatter):
    """Log formatter that writes one JSON object per line.

    Usage::

        handler.setFormatter(JsonFormatter())
    """

    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "level": record.levelname,
            "module": record.name,
            "message": mask_api_key(record.getMessage()),
        }
        return json.dumps(log_dict, ensure_ascii=False)
