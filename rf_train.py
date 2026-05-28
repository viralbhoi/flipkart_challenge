import pandas as pd
import joblib
import os

# 0. FOLDER SETUP
# Create the 'demand' folder if it doesn't exist
output_dir = 'demand'
os.makedirs(output_dir, exist_ok=True)

# 1. Load the data
train = pd.read_csv('cleanedA/Train_Standard_Models.csv')
test = pd.read_csv('cleanedA/Test_Standard_Models.csv')

X_train = train.drop(columns=['demand'])
y_train = train['demand']
X_test = test.drop(columns=['Index'])

# 2. CHOOSE YOUR MODEL
# (Random Forest):
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
model_name = "rf"

# 3. Train and Predict
print(f"Training {model_name.upper()} model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 4. Save outputs directly into the 'demand' folder!
csv_filename = os.path.join(output_dir, f"{model_name}_result.csv")
pkl_filename = os.path.join(output_dir, f"{model_name}_model.pkl")

pd.DataFrame({'Index': test['Index'], 'demand': predictions}).to_csv(csv_filename, index=False)
joblib.dump(model, pkl_filename)

print(f"Done! Saved CSV perfectly formatted to: {csv_filename}")