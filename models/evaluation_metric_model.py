from utils.database import db

class EvaluationMetric(db.Model):
    __tablename__ = 'evaluation_metric'
    
    metric_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model_info.model_id'), nullable=False)
    # 原有核心字段
    accuracy = db.Column(db.DECIMAL(10,4), default=0.0000, comment='调整兰德指数(ARI)')
    loss_value = db.Column(db.DECIMAL(10,4), default=0.0000, comment='损失值')
    # 新增无监督评估指标字段（已同步到表）
    silhouette_score = db.Column(db.DECIMAL(10,4), default=0.0000, comment='轮廓系数')
    dbi_score = db.Column(db.DECIMAL(10,4), default=9999.99, comment='DBI指数')
    ch_score = db.Column(db.DECIMAL(10,4), default=0.0000, comment='CH指数')
    
    # 新增增广数据集评估标准差字段（请先执行database/update_std_fields.sql添加字段后再取消注释）
    accuracy_std = db.Column(db.DECIMAL(10,4), default=0.0000, comment='ARI标准差')
    silhouette_std = db.Column(db.DECIMAL(10,4), default=0.0000, comment='轮廓系数标准差')
    dbi_std = db.Column(db.DECIMAL(10,4), default=0.0000, comment='DBI指数标准差')
    ch_std = db.Column(db.DECIMAL(10,4), default=0.0000, comment='CH指数标准差')
    
    # 关联模型信息
    model = db.relationship('ModelInfo', backref=db.backref('metrics', lazy=True))
