from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

# Carga el modelo

# Enruta la landing page (endpoint /)

# Enruta la funcion al endpoint /api/v1/predict

# Enruta la funcion al endpoint /api/v1/retrain


if __name__ == '__main__':
    app.run(debug=True)
