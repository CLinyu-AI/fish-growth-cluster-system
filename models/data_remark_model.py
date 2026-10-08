from utils.database import db
from datetime import datetime

class DataRemark(db.Model):
    __tablename__ = 'data_remark'
    
    remark_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    remark_type = db.Column(db.String(50), nullable=False, comment='备注关联类型：dataset/模型model')
    relation_id = db.Column(db.Integer, nullable=False, comment='关联主体ID，对应dataset_id/model_id')
    remark_content = db.Column(db.Text, nullable=False, comment='备注详情内容')
    is_active = db.Column(db.Boolean, default=True, comment='是否对普通用户可见')
    creator_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)