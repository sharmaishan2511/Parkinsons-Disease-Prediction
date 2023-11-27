import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

import joblib

application = Flask(__name__)

model = joblib.load('artifacts/model.pkl')

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        fo_hz = float(request.form.get('MDVPFoHz'))
        fhi_hz = float(request.form.get('MDVPFhiHz'))
        flo_hz = float(request.form.get('MDVPFloHz'))
        hnr = float(request.form.get('HNR'))
        spread1 = float(request.form.get('spread1'))
        d2 = float(request.form.get('D2'))

        input_data = np.array([fo_hz, fhi_hz, flo_hz, hnr, spread1, d2]).reshape(1, -1)

        result = model.predict(input_data)
        print(result)
        return render_template('home.html', result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
