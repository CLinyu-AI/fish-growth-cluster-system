from utils.database import db
from datetime import datetime

class ModelInfo(db.Model):
    __tablename__ = 'model_info'
    model_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(50), nullable=False)  # 'kmeans' / 'dbscan'
    model_path = db.Column(db.String(200), nullable=False)
    scaler_path = db.Column(db.String(200), nullable=False)  # 标准化器路径
    pca_path = db.Column(db.String(200), nullable=True)  # PCA降维模型路径，为空表示未启用PCA
    is_deployed = db.Column(db.Boolean, default=False)  # 是否部署到前端
    create_time = db.Column(db.DateTime, default=datetime.now)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.dataset_id'), nullable=False)
    # 新增字段后需执行数据库更新SQL：
    # ALTER TABLE model_info ADD COLUMN pca_path VARCHAR(200) NULL COMMENT 'PCA降维模型路径' AFTER scaler_path;
    # remark = db.Column(db.Text, default='', nullable=True)  # 模型备注信息，需先在数据库执行ALTER TABLE添加此字段
