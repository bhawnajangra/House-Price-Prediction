from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    area = float(request.form['area'])
    mainroad = int(request.form['mainroad'])
    furnishingstatus = int(request.form['furnishingstatus'])
    basement = int(request.form['basement'])

    features = np.array([[area, mainroad, furnishingstatus, basement]])

    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Predicted House Price: ₹ {output}'
    )

if __name__ == "__main__":
    app.run(debug=True)