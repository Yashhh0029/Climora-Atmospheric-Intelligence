import os
from flask import Flask
from flask_cors import CORS
from config import Config
from utils.logger import logger
from routes import register_routes
from middleware import register_middleware

def create_app():
    # Use absolute paths for templates and static files to point to the frontend directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(base_dir, 'frontend')
    template_dir = os.path.join(frontend_dir, 'templates')
    static_dir = os.path.join(frontend_dir, 'static')
    
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config.from_object(Config)
    CORS(app)
    
    os.makedirs(app.config.get('CACHE_DIR', 'model_cache'), exist_ok=True)
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'dataset'), exist_ok=True)

    register_middleware(app)
    register_routes(app)
    
    return app

app = create_app()

if __name__ == '__main__':
    port = app.config.get('PORT', 5000)
    try:
        from waitress import serve
        logger.info(f"Starting production server with Waitress on port {port}...")
        serve(app, host='0.0.0.0', port=port)
    except ImportError:
        logger.info(f"Waitress not installed. Falling back to Flask development server on port {port}...")
        app.run(host='0.0.0.0', debug=app.config.get('DEBUG', True), port=port)
