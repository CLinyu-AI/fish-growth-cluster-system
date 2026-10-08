from utils.database import db
from datetime import datetime

class PredictRecord(db.Model):
    """预测记录表"""
    __tablename__ = 'predict_records'
    
    record_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False, comment='用户ID')
    model_id = db.Column(db.Integer, db.ForeignKey('model_info.model_id'), nullable=False, comment='使用的模型ID')
    model_name = db.Column(db.String(100), nullable=False, comment='模型名称')
    length = db.Column(db.Float, nullable=False, comment='输入的长度(cm)')
    weight = db.Column(db.Float, nullable=False, comment='输入的重量(g)')
    w_l_ratio = db.Column(db.Float, nullable=False, comment='输入的重长比')
    predicted_label = db.Column(db.Integer, nullable=False, comment='预测的聚类标签')
    predicted_species = db.Column(db.String(50), nullable=False, comment='预测的鱼类类别')
    confidence = db.Column(db.Float, comment='预测置信度(%)')
    predict_time = db.Column(db.DateTime, default=datetime.now, comment='预测时间')
    
    # 关联关系
    user = db.relationship('User', backref=db.backref('predict_records', lazy=True))
    model = db.relationship('ModelInfo', backref=db.backref('predict_records', lazy=True))
    
    def to_dict(self):
        return {
            'record_id': self.record_id,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'length': self.length,
            'weight': self.weight,
            'w_l_ratio': self.w_l_ratio,
            'predicted_label': self.predicted_label,
            'predicted_species': self.predicted_species,
            'confidence': self.confidence,
            'predict_time': self.predict_time.strftime('%Y-%m-%d %H:%M:%S')
        }