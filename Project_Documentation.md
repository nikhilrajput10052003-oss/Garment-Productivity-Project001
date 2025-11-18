Project Documentation: Productivity Prediction
1. Project Objective
The objective is to build a machine learning model to predict the productivity of garment worker teams. The model is then deployed as a web application using Flask.
2. Data
Source: Kaggle's "Productivity Prediction of Garment Employees" dataset.
Link: https://www.kaggle.com/datasets/utkarshsarbahi/productivity-prediction-of-garment-employees
3. Part 1: Model Training (Training files/Employee_Prediction.py)
Load Data: The garments_worker_productivity.csv is loaded.
Pre-processing:
The wip column is dropped due to many missing values.
The date column is converted to a month feature.
The department column is cleaned of whitespace.
Encoding: OrdinalEncoder is used to convert quarter, department, and day columns into numbers so the model can understand them.
Modeling: An XGBoost Regressor is trained on the data (using parameters from the internship guide, e.g., n_estimators=200).
Output: The script saves the trained model as gwp.pkl.
4. Part 2: Web Application (Flask/app.py)
Load Model: The Flask app loads the gwp.pkl file on startup.
Routing:
/: Renders home.html.
/about: Renders about.html.
/predict: Renders predict.html, which contains the input form.
Prediction Logic (/submit):
The app receives the 13 numeric inputs from the form.
It passes these 13 numbers to the loaded model for prediction.
It uses if/elif/else logic (as per the internship screenshots) to classify the numeric prediction into "medium productive," or "Highly productive."

This final text is displayed on the submit.html page.
