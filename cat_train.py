import pandas as pd
from catboost import CatBoostRegressor
import joblib
import os

# ==========================================
# 0. FOLDER SETUP
# ==========================================
# Create the 'demand' folder if it doesn't exist
output_dir = 'demand'
os.makedirs(output_dir, exist_ok=True)

# 1. Load the data (from Pipeline B)
train = pd.read_csv('cleanedB/Train_CatBoost.csv')
test = pd.read_csv('cleanedB/Test_CatBoost.csv')

X_train = train.drop(columns=['demand'])
y_train = train['demand']
X_test = test.drop(columns=['Index'])

# 2. Define text columns for CatBoost
categorical_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']

# 3. Initialize Model
model_name = "catboost"
model = CatBoostRegressor(
    iterations=500, 
    learning_rate=0.05, 
    cat_features=categorical_features, 
    verbose=100
)

# 4. Train and Predict
print(f"Training {model_name.upper()} model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 5. Save outputs directly into the 'demand' folder!
csv_filename = os.path.join(output_dir, f"{model_name}_result.csv")
pkl_filename = os.path.join(output_dir, f"{model_name}_model.pkl")

pd.DataFrame({'Index': test['Index'], 'demand': predictions}).to_csv(csv_filename, index=False)
joblib.dump(model, pkl_filename)

print(f"Done! Saved CSV perfectly formatted to: {csv_filename}")