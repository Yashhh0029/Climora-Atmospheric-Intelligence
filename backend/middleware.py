import time
from flask import request, jsonify, g
from utils.logger import logger
from utils.metrics import metrics_lock, system_metrics
import uuid

rate_limit_store = {}

def register_middleware(app):
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.request_id = str(uuid.uuid4())
        
        ip = request.remote_addr
        now = time.time()
        if ip not in rate_limit_store:
            rate_limit_store[ip] = []
        rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < 60]
        
        RATE_LIMIT = app.config.get('RATE_LIMIT', 30)
        if len(rate_limit_store[ip]) >= RATE_LIMIT:
            logger.warning(f"Rate limit exceeded for IP: {ip}", extra={'event': 'rate_limit_exceeded'})
            return jsonify({"error": "Too Many Requests"}), 429
        rate_limit_store[ip].append(now)

    @app.after_request
    def after_request(response):
        if request.path in ['/predict', '/', '/health', '/metrics']:
            duration = time.time() - getattr(g, 'start_time', time.time())
            with metrics_lock:
                system_metrics['total_requests'] += 1
                system_metrics['total_response_time'] += duration
            
            logger.info(
                f"{request.method} {request.path}",
                extra={'event': 'request_end', 'duration': round(duration, 4), 'status': response.status_code}
            )
        return response
