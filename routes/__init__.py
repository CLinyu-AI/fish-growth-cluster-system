from routes.auth_routes import auth_bp
from routes.admin_routes import admin_bp
from routes.user_routes import user_bp

__all__ = ['auth_bp', 'admin_bp', 'user_bp']