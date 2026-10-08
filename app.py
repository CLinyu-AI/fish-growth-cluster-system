import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, render_template, request, redirect, url_for, session
from config import Config
from utils.database import db
from routes import auth_bp, admin_bp, user_bp

# 初始化Flask应用
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)
db.init_app(app)

# 注册蓝图
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(user_bp, url_prefix='/user')


# 首页路由（重定向到管理员/用户面板）
@app.route('/')
def index():
    if 'user_id' in session:
        if session['role'] == 'admin':
            return redirect(url_for('admin.dashboard'))
        else:
            return redirect(url_for('user.dashboard'))
    return redirect(url_for('auth.login'))

# 注册404错误处理器
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# 注册500错误处理器
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# 启动应用
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 确保表存在
    app.run(debug=True, host='0.0.0.0', port=5000)