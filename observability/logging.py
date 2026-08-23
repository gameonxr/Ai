from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonLineFormatter(logging.Formatter):
    """Format log records as one machine-readable JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"level": record.levelname, "logger": record.name, "message": record.getMessage(), "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z")}
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "data"):
            payload["data"] = record.data
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO", file_path: str | Path | None = None) -> logging.Logger:
    """Configure the simulator logger idempotently and return it."""
    logger = logging.getLogger("ai_body_simulator")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
    if file_path is not None:
        output = Path(file_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        matching = next((handler for handler in logger.handlers if getattr(handler, "_ai_file", False) and Path(getattr(handler, "baseFilename", "")).resolve() == output), None)
        if matching is None or not output.exists():
            for existing in list(logger.handlers):
                if getattr(existing, "_ai_file", False):
                    logger.removeHandler(existing)
                    existing.close()
            handler = logging.FileHandler(output, encoding="utf-8")
            handler._ai_file = True
            handler.setFormatter(JsonLineFormatter())
            logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, level: int, event: str, data: dict | None = None) -> None:
    logger.log(level, event, extra={"event": event, "data": data or {}})
