import os

class Config:
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("FLASK_SECRET_KEY environment variable is strictly required for production deployment.")
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 5 * 1024 * 1024)) # 5 MB upload limit
    RATE_LIMIT = int(os.environ.get('RATE_LIMIT', 30))
    MODEL_VERSION = os.environ.get('MODEL_VERSION', "6.0")
    CACHE_DIR = "model_cache"
    UPLOAD_FOLDER = 'dataset'
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
