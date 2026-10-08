from flask import Blueprint, render_template, request, redirect, url_for, session, make_response
from services.auth_service import login_user, register_user
from utils.captcha import generate_captcha_code, generate_captcha_image, verify_captcha

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 1. 验证验证码
        input_captcha = request.form.get('captcha', '')
        if not verify_captcha(input_captcha):
            return render_template('login.html', error='验证码错误')
        
        # 2. 验证账号密码（适配新的返回值）
        username = request.form['username']
        password = request.form['password']
        user, role, error_msg = login_user(username, password)
        
        if user:
            session.pop('captcha', None)
            session['user_id'] = user.user_id
            session['username'] = user.username  # 新增这行
            session['role'] = role
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error=error_msg)
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role_type = request.form['role_type']
        invite_code = request.form.get('invite_code', '')  # 新增邀请码接收
        
        success, message = register_user(username, password, confirm_password, role_type, invite_code)
        if success:
            return render_template('register.html', success=message)
        else:
            return render_template('register.html', error=message)
    return render_template('register.html')

@auth_bp.route('/captcha')
def captcha():
    """生成验证码图片路由"""
    code = generate_captcha_code()
    session['captcha'] = code
    image_buf = generate_captcha_image(code)
    response = make_response(image_buf.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))