from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.pkl')
le_make, le_model, le_fuel, le_trans = joblib.load('encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    make = le_make.transform([data['make']])[0]
    model_ = le_model.transform([data['model']])[0]
    year = int(data['year'])
    engine = float(data['engine_size'])
    mileage = float(data['mileage'])
    fuel = le_fuel.transform([data['fuel_type']])[0]
    trans = le_trans.transform([data['transmission']])[0]

    X = np.array([[make, model_, year, engine, mileage, fuel, trans]])
    prediction = model.predict(X)[0]

    result = "Expensive" if prediction == 1 else "Affordable"
    return render_template('index.html', prediction_text=f"Prediction: This car is likely {result}")

if __name__ == "__main__":
    app.run(debug=True)
