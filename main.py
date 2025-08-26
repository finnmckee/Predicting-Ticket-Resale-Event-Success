# Adjusted script using the exact column names from your spreadsheet

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIG ===
input_file = "C:/Users/fishk/Downloads/Working AutoIQ Concert Data.xlsx"
sheet_name = "Sheet3"
output_predictions = "C:/Users/fishk/Downloads/xgboost_resale_predictions_working.xlsx"

# === Load Excel File ===
df = pd.read_excel(input_file, sheet_name=sheet_name)

# === Preprocessing ===
# Set the exact target column name
target_col = "Simple Resale Potential Score"

# Drop rows missing the target
df = df.dropna(subset=[target_col])

# Select relevant features from available columns
features = [
    "TN Get In",
    "Total Venue Ticket Quantity (TN)",
    "Days Until Event as of 5/12/25",
    "Is Weekend"
]

# Ensure features exist in dataframe
features = [feature for feature in features if feature in df.columns]

X = df[features]
y = df[target_col]

# Normalize numeric features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Train XGBoost Regressor ===
model = XGBRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === Feature Importances ===
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

importances = importances.sort_values(ascending=False)
print("Feature Importances:\n")
print(importances)

# === Save Predictions ===
predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
predictions_df.to_excel(output_predictions, index=False)

# Display the top of the predictions DataFrame
print(predictions_df.head())

# Or to view all rows:
print(predictions_df.to_string())

predictions_df.to_excel("xgboost_resale_predictions.xlsx", index=False)

importances_df = importances.reset_index()
importances_df.columns = ['Feature', 'Importance']
importances_df.to_excel("xgboost_feature_importance.xlsx", index=False)
