import joblib
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')
label_gender = joblib.load('LabelEncoderGender.pkl')
label_geography = joblib.load('LabelEncoderGeography.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
            'CreditScore': [int(request.form['CreditScore'])],
            'Geography': [request.form['Geography']],
            'Gender': [request.form['Gender']],
            'Age': [int(request.form['Age'])],
            'Tenure': [int(request.form['Tenure'])],
            'Balance': [float(request.form['Balance'])],
            'NumOfProducts': [int(request.form['NumOfProducts'])],
            'HasCrCard': [int(request.form['HasCrCard'])],
            'IsActiveMember': [int(request.form['IsActiveMember'])],
            'EstimatedSalary': [float(request.form['EstimatedSalary'])]
        }
    
    try:
        input_data['Gender'] = label_gender.transform(input_data['Gender'])
    except ValueError:
        input_data['Gender'] = label_gender.transform(['Unknown'])[0]

    try:
        input_data['Geography'] = label_geography.transform(input_data['Geography'])
    except ValueError:
        input_data['Geography'] = label_geography.transform(['Unknown'])[0]

    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
