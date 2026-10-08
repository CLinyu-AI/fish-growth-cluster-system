import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:Cjl19525340080?@localhost:3306/fish_system?charset=utf8mb4'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 文件保存路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_UPLOADS_DIR = os.path.join(BASE_DIR, 'data_uploads')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    
    # 管理员注册邀请码（可自定义修改）
    ADMIN_INVITE_CODE = "fish_admin_2026"

# 自动创建文件夹
os.makedirs(Config.DATA_UPLOADS_DIR, exist_ok=True)
os.makedirs(Config.SAVED_MODELS_DIR, exist_ok=True)