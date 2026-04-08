from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = 'prediction_events'
    id = db.Column(db.Integer, primary_key=True)
    tv = db.Column(db.Float)
    radio = db.Column(db.Float)
    newspaper = db.Column(db.Float)
    prediction = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "tv": self.tv,
            "radio": self.radio,
            "newspaper": self.newspaper,
            "prediction": self.prediction,
            "created_at": self.created_at.isoformat()
        }