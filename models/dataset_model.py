from utils.database import db
from datetime import datetime

class Dataset(db.Model):
    __tablename__ = 'dataset'
    dataset_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.now)
    uploader_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)