import re
from models import User, UserRole
from utils.database import db
from werkzeug.security import check_password_hash, generate_password_hash
from config import Config

# 账号格式正则
ADMIN_USERNAME_PATTERN = r'^admin\d+$'  # admin开头+数字
USER_USERNAME_PATTERN = r'^\d+$'        # 纯数字

def validate_username(username, role_type):
    """校验账号格式是否符合角色要求"""
    if role_type == 'admin':
        if not re.match(ADMIN_USERNAME_PATTERN, username):
            return False, "管理员账号必须以admin开头，后跟数字（如admin123）"
    elif role_type == 'user':
        if not re.match(USER_USERNAME_PATTERN, username):
            return False, "普通用户账号必须为纯数字（如123456）"
    return True, ""

def login_user(username, password):
    """加密登录验证（自动识别角色格式）"""
    # 先校验账号格式
    if re.match(ADMIN_USERNAME_PATTERN, username):
        role_name = 'admin'
    elif re.match(USER_USERNAME_PATTERN, username):
        role_name = 'user'
    else:
        return None, None, "账号格式错误（管理员：admin+数字 | 用户：纯数字）"
    
    # 验证账号密码
    user = User.query.filter_by(username=username).first()
    if not user:
        return None, None, "账号不存在"
    if not check_password_hash(user.password, password):
        return None, None, "密码错误"
    
    # 验证角色匹配
    user_role = UserRole.query.get(user.role_id)
    if user_role.role_name != role_name:
        return None, None, "账号角色与格式不匹配"
    
    return user, role_name, ""

def register_user(username, password, confirm_password, role_type, invite_code=None):
    """用户注册逻辑（新增管理员邀请码校验）"""
    # 1. 基础校验
    if password != confirm_password:
        return False, "两次输入的密码不一致"
    if len(password) < 6:
        return False, "密码长度不能少于6位"
    
    # 2. 管理员邀请码校验
    if role_type == 'admin':
        if not invite_code or invite_code.strip() != Config.ADMIN_INVITE_CODE:
            return False, "管理员邀请码错误，无法注册管理员账号"
    
    # 3. 校验账号格式
    is_valid, msg = validate_username(username, role_type)
    if not is_valid:
        return False, msg
    
    # 4. 校验用户名是否已存在
    if User.query.filter_by(username=username).first():
        return False, "账号已存在，请更换账号"
    
    # 5. 获取/创建角色
    role = UserRole.query.filter_by(role_name=role_type).first()
    if not role:
        role = UserRole(role_name=role_type)
        db.session.add(role)
        db.session.commit()
    
    # 6. 加密密码，创建新用户
    hashed_password = generate_password_hash(password)
    new_user = User(
        username=username,
        password=hashed_password,
        role_id=role.role_id
    )
    
    db.session.add(new_user)
    db.session.commit()
    return True, f"{role_type}账号注册成功！请点击返回登录"