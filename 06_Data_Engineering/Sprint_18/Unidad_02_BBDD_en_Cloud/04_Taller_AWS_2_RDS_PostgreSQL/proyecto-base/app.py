from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from dotenv import load_dotenv
from models import db, Prediction
import click

load_dotenv()
os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.cli.command("init-db")
def init_db():
    db.create_all()
    click.echo("Tablas creadas.")

with open('ad_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API del modelo advertising"

@app.route("/api/v1/predict", methods=["GET"])
def predict():
    tv = request.args.get('tv', np.nan, type=float)
    radio = request.args.get('radio', np.nan, type=float)
    newspaper = request.args.get('newspaper', np.nan, type=float)

    missing = [name for name, val in [('tv', tv), ('radio', radio), ('newspaper', newspaper)] if np.isnan(val)]
    input_data = pd.DataFrame({'tv': [tv], 'radio': [radio], 'newspaper': [newspaper]})
    result = round(float(model.predict(input_data)[0]), 1)

    record = Prediction(
        tv = None if np.isnan(tv) else round(tv, 1),
        radio = None if np.isnan(radio) else round(radio, 1),
        newspaper = None if np.isnan(newspaper) else round(newspaper, 1),
        prediction = result
    )
    try:
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error guardando en BD: {e}")

    response = {'predictions': result}
    if missing:
        response['warning'] = f"Missing values imputed for: {', '.join(missing)}"
    return jsonify(response)

@app.route("/api/v1/predictions", methods=["GET"])
def get_predictions():
    limit = request.args.get('limit', 50, type=int)
    records = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).limit(limit).all()
    return jsonify([r.to_dict() for r in records])

@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    global model
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')
        data.columns = [col.lower() for col in data.columns]
        X = data.drop(columns=['sales'])
        y = data['sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(X, y)
        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == '__main__':
    app.run(debug=True)