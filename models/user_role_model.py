from utils.database import db

class UserRole(db.Model):
    __tablename__ = 'user_role'
    role_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    role_name = db.Column(db.String(20), unique=True, nullable=False)  # 'admin' / 'user'