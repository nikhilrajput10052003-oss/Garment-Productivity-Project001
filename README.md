Garment Worker Productivity Prediction

This project uses an XGBoost machine learning model to predict the productivity of garment worker teams. The model is trained on a Kaggle dataset and deployed using a Flask web application.

This project follows the structure and requirements of the virtual ML internship.

How to Run

Install Packages:

pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask


Get Dataset:

Download garments_worker_productivity.csv from Kaggle.

Place it inside the Training files/ folder.

Train Model:

Navigate to the Training files folder in your terminal.

Run the script to create the model file:

cd "Training files"
python Employee_Prediction.py


Move Model:

Go to the Training files folder.

Copy the new gwp.pkl file.

Paste it into the Flask/ folder.

Run Web App:

Navigate to the Flask folder in your terminal.

Run the app:

cd .. 
cd Flask
python app.py


View Project:

Open your web browser and go to http://127.0.0.1:5000/