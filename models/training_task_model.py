from utils.database import db
from datetime import datetime

class TrainingTask(db.Model):
    __tablename__ = 'training_task'
    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.dataset_id'), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='pending')  # 'pending' / 'running' / 'completed'
    create_time = db.Column(db.DateTime, default=datetime.now)