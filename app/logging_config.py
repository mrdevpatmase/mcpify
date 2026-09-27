"""
Structured (JSON) logging setup. Plain text log lines
("%(asctime)s [%(levelname)s] ...") read fine in a local terminal but
are awkward for a log aggregator (Render's log stream, CloudWatch,
Datadog, whatever) to filter/query on - "show me every ERROR from
app.proxy in the last hour" needs a parser for free-text, but is a
field lookup against JSON lines. Every existing logger.info(...)/
logger.warning(...) call site anywhere in this app is untouched by
this: only the formatter attached to the root handler changes, so this
needed no changes anywhere else.
"""
import json
import logging
import os


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """
    LOG_FORMAT=json (the default) emits one JSON object per line;
    LOG_FORMAT=text keeps the original human-readable format for local
    dev, since a terminal full of JSON objects is worse to eyeball
    while actively debugging than the plain line was.
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler()
    if log_format == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    else:
        handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
