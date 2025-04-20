from flask import Flask, request, jsonify, render_template # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


appliaction = Flask(__name__)
app = appliaction


ridge_model = pickle.load(open('models/ridge_regression.pkl', 'rb'))

standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET', 'POST'])
def  predict_datapoint():
    if request.method == 'POST':

        Temparature = request.form.get('Temparature')
        RH = request.form.get('RH')
        WS = request.form.get('WS')
        Rain = request.form.get('Rain')
        FFMC = request.form.get('FFMC')
        DMC = request.form.get('DMC')
        ISI = request.form.get('ISI')
        Classes = request.form.get('Classes')
        Region = request.form.get('Region')

        new_data_scaled = standard_scaler.transform([[Temparature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])

        result = round(ridge_model.predict(new_data_scaled)[0], 2)

        return render_template('home.html', result = result)

    else:

        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)