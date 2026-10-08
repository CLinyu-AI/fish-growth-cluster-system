from utils.database import db
from datetime import datetime

class AdminOperationLog(db.Model):
    __tablename__ = 'admin_operation_log'
    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    operation = db.Column(db.String(100), nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)