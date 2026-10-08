from utils.database import db

class Result(db.Model):
    __tablename__ = 'result'
    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_record_id = db.Column(db.Integer, db.ForeignKey('prediction_record.record_id'), nullable=False)
    predicted_species = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)