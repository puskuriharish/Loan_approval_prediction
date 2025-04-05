from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        features = [float(request.form[key]) for key in request.form]
        
        # Convert input to NumPy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(input_data)
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)