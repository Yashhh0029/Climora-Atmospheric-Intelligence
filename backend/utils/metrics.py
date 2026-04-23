import threading

training_lock = threading.Lock()
metrics_lock = threading.Lock()

system_metrics = {
    'total_requests': 0,
    'cache_hits': 0,
    'model_trainings': 0,
    'total_response_time': 0.0
}
