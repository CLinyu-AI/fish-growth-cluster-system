from models import User, UserRole
from utils.database import db
from werkzeug.security import generate_password_hash

def get_all_users():
    """获取所有用户列表"""
    return User.query.join(UserRole).add_columns(
        User.user_id, User.username, UserRole.role_name
    ).all()

def create_user(username, password, role_id):
    """创建新用户（密码加密）"""
    if User.query.filter_by(username=username).first():
        return False, "用户名已存在"
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password, role_id=role_id)
    db.session.add(new_user)
    db.session.commit()
    return True, "用户创建成功"

def delete_user(user_id):
    """删除用户"""
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return True, "用户删除成功"
    return False, "用户不存在"

def update_user_role(user_id, new_role_id):
    """修改用户角色"""
    user = User.query.get(user_id)
    if user:
        user.role_id = new_role_id
        db.session.commit()
        return True, "角色修改成功"
    return False, "用户不存在"