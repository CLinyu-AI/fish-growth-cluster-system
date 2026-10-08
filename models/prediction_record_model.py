from utils.database import db
from datetime import datetime

class PredictionRecord(db.Model):
    __tablename__ = 'prediction_record'
    record_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model_info.model_id'), nullable=False, comment='使用的模型ID')
    model_name = db.Column(db.String(100), nullable=False, comment='模型名称')
    input_length = db.Column(db.Float, nullable=False, comment='输入的长度(cm)')
    input_weight = db.Column(db.Float, nullable=False, comment='输入的重量(g)')
    input_l_w_ratio = db.Column(db.Float, nullable=False, comment='输入的长重比')
    predicted_label = db.Column(db.Integer, nullable=False, comment='预测的聚类标签')
    predicted_species = db.Column(db.String(50), nullable=False, comment='预测的鱼类类别')
    confidence = db.Column(db.Float, comment='预测置信度(%)')
    create_time = db.Column(db.DateTime, default=datetime.now, comment='预测时间')
    
    # 关联关系
    user = db.relationship('User', backref=db.backref('prediction_records', lazy=True))
    model = db.relationship('ModelInfo', backref=db.backref('prediction_records', lazy=True))
