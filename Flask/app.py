# This is the "Web App" script.

import numpy as np
from flask import Flask, request, render_template
import pickle
import warnings

warnings.filterwarnings('ignore')

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("gwp.pkl", "rb"))
    print("Model 'gwp.pkl' loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'gwp.pkl' not found in 'Flask' folder.")
    print("Please run the training script first, then copy 'gwp.pkl' here.")
    exit()

# Route for the home page
@app.route("/")
def home():
    return render_template("home.html")

# Route for the about page
@app.route("/about")
def about():
    return render_template("about.html")

# Route to show the prediction input form
@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# Route to handle the prediction
@app.route("/submit", methods=["POST"])
def submit():
    try:
        form_data = request.form
        
        # Create the feature list in the exact order the model was trained on
        # This matches the screenshot 'image_2fd561.png'
        feature_list = [
            float(form_data['quarter']),
            float(form_data['department']),
            float(form_data['day']),
            float(form_data['team']),
            float(form_data['targeted_productivity']),
            float(form_data['smv']),
            float(form_data['over_time']),
            float(form_data['incentive']),
            float(form_data['idle_time']),
            float(form_data['idle_men']),
            float(form_data['no_of_style_change']),
            float(form_data['no_of_workers']),
            float(form_data['month'])
        ]
        
        # Make prediction
        prediction = model.predict([feature_list])
        prediction_value = prediction[0]
        
        # This is the logic from your screenshot 'image_2fd561.png'
        if prediction_value < 0.3:
            text = 'The employee is averagely productive.'
        elif prediction_value >= 0.3 and prediction_value < 0.8:
            text = 'The employee is medium productive.'
        else:
            text = 'The employee is Highly productive.'

        return render_template("submit.html", prediction_text=text)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("submit.html", prediction_text=f"Error: Could not process input. {e}")

# Main function to run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)