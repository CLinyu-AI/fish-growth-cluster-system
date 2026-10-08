from utils.database import db

class FishData(db.Model):
    __tablename__ = 'fish_data'
    data_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    species = db.Column(db.String(50))  # 真实标签
    length = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    l_w_ratio = db.Column(db.Float, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.dataset_id'), nullable=False)