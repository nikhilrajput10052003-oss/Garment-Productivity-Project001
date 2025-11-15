# This is the "Model Factory".
# Its only job is to create the "gwp.pkl" model file.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.metrics import r2_score
import pickle
import warnings

warnings.filterwarnings('ignore')

print("--- Model Training Script Started ---")

# --- 1. Load Data ---
try:
    data = pd.read_csv("garments_worker_productivity.csv")
    print("Dataset 'garments_worker_productivity.csv' loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'garments_worker_productivity.csv' not found.")
    print("Please download it from Kaggle and place it in this 'Training files' folder.")
    exit()

# --- 2. Pre-processing (as per your instructions) ---
print("Pre-processing data...")
data.drop(['wip'], axis=1, inplace=True)
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data.drop(['date'], axis=1, inplace=True)
data['department'] = data['department'].str.strip()

# --- 3. Encode Categorical Data ---
# We do this to train the model properly.
cat_cols = ['quarter', 'department', 'day']
encoder = OrdinalEncoder()
data[cat_cols] = encoder.fit_transform(data[cat_cols])
print("Data encoded.")

# --- 4. Split Data ---
X = data.drop(['actual_productivity'], axis=1)
y = data['actual_productivity']

# This prints the final order of columns, which we need for the app
print(f"Final column order for model: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Model Building (using your screenshot's parameters) ---
print("Training XGBoost model...")
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', 
                             n_estimators=200, 
                             max_depth=5, 
                             learning_rate=0.1, 
                             random_state=42)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

print(f"Model trained. R2 Score: {r2_score(y_test, pred_xgb)}")

# --- 6. Save the Model ---
# This is the most important step.
pickle.dump(model_xgb, open("gwp.pkl", "wb"))
print("Model saved as 'gwp.pkl' in 'Training files' folder.")

print("\n--- Model Training Script Finished ---")