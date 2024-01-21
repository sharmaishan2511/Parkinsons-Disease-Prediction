import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import joblib

application = Flask(__name__)

model = joblib.load('model.pkl')

app = application

@app.route('/', methods=['GET', 'POST'])
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

        # Create a DataFrame from the input data
        input_df = pd.DataFrame({
            'MDVPFoHz': [fo_hz],
            'MDVPFhiHz': [fhi_hz],
            'MDVPFloHz': [flo_hz],
            'HNR': [hnr],
            'spread1': [spread1],
            'D2': [d2]
        })

        # Use the loaded scaler to transform the input data
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)

        result = model.predict(input_scaled)
        if result == 1:
            result_message = "The parameters indicate Parkinson's Disease."
        else:
            result_message = "The parameters do not indicate Parkinson's Disease."

        return render_template('home.html', result=result_message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
