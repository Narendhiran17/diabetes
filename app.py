from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("Lr2.pkl", "rb"))
scaler = pickle.load(open("Scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
