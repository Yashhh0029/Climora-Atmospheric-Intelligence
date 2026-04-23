import logging
import json
from datetime import datetime
import os
from flask import g

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_record['duration'] = record.duration
        if hasattr(record, 'event'):
            log_record['event'] = record.event
        if hasattr(record, 'status'):
            log_record['status'] = record.status
        return json.dumps(log_record)

class ContextFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = getattr(g, 'request_id', 'system')
        except RuntimeError:
            record.request_id = 'system'
        return True

def setup_logger():
    log = logging.getLogger('climora')
    log.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    ch = logging.StreamHandler()
    ch.setFormatter(JsonFormatter())
    log.addHandler(ch)
    log.addFilter(ContextFilter())
    return log

logger = setup_logger()
