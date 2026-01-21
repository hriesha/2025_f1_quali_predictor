import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Bahrain GP data
session_fp3_2024 = fastf1.get_session(2024, "Bahrain", "FP3")
session_fp3_2024.load()

session_quali_2024 = fastf1.get_session(2024, "Bahrain", "Q")
session_quali_2024.load()

# Extract FP3 lap times and sector times
fp3_laps = session_fp3_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
fp3_laps.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    fp3_laps[f"{col} (s)"] = fp3_laps[col].dt.total_seconds()

# Calculate average sector performance
avg_sectors = fp3_laps.groupby("Driver").agg({
    "LapTime (s)": "mean",
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

# Get 2024 qualifying positions
quali_results = session_quali_2024.results[["Abbreviation", "Position"]].copy()
quali_results.columns = ["Driver", "QualiPosition"]

# 2025 FP3 Data (Bahrain)
fp3_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "SAI", "ALO", "GAS", "TSU", "HUL", "OCO"],
    "FP3Time (s)": [89.5, 89.6, 89.7, 89.8, 89.9, 90.0, 90.1, 90.2, 90.3, 90.4, 90.6, 90.7],
    "Sector1Time (s)": [28.1, 28.2, 28.3, 28.2, 28.4, 28.5, 28.5, 28.6, 28.7, 28.6, 28.8, 28.9],
    "Sector2Time (s)": [36.2, 36.3, 36.3, 36.4, 36.4, 36.5, 36.6, 36.5, 36.7, 36.8, 36.9, 37.0],
    "Sector3Time (s)": [25.2, 25.1, 25.1, 25.2, 25.1, 25.0, 25.0, 25.1, 24.9, 25.0, 24.9, 24.8]
})

# Team performance based on current season
team_strength = {
    "McLaren": 1.00, "Red Bull": 0.95, "Mercedes": 0.90, "Ferrari": 0.88,
    "Alpine": 0.82, "Aston Martin": 0.85, "Racing Bulls": 0.80, "Haas": 0.78
}

driver_teams = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
    "RUS": "Mercedes", "HAM": "Mercedes", "SAI": "Ferrari", "ALO": "Aston Martin",
    "GAS": "Alpine", "TSU": "Racing Bulls", "HUL": "Haas", "OCO": "Alpine"
}

fp3_2025["Team"] = fp3_2025["Driver"].map(driver_teams)
fp3_2025["TeamStrength"] = fp3_2025["Team"].map(team_strength)

# Driver experience factor (years in F1)
driver_experience = {
    "VER": 10, "NOR": 6, "PIA": 2, "LEC": 7, "RUS": 6, "HAM": 17,
    "SAI": 10, "ALO": 22, "GAS": 8, "TSU": 4, "HUL": 13, "OCO": 8
}

fp3_2025["Experience"] = fp3_2025["Driver"].map(driver_experience)
fp3_2025["ExperienceScore"] = fp3_2025["Experience"] / 22  # Normalize

# Night race performance factor
night_performance = {
    "VER": 1.02, "NOR": 0.98, "PIA": 0.97, "LEC": 1.01, "RUS": 0.99,
    "HAM": 1.03, "SAI": 1.00, "ALO": 1.02, "GAS": 0.96, "TSU": 0.95,
    "HUL": 0.98, "OCO": 0.97
}

fp3_2025["NightFactor"] = fp3_2025["Driver"].map(night_performance)

# Merge 2024 data for training
merged_data = avg_sectors.merge(quali_results, on="Driver", how="inner")
merged_data["TeamStrength"] = merged_data["Driver"].map(driver_teams).map(team_strength)
merged_data["ExperienceScore"] = merged_data["Driver"].map(driver_experience).map(lambda x: x / 22 if pd.notna(x) else 0.5)
merged_data["NightFactor"] = merged_data["Driver"].map(night_performance)

# Prepare training data
feature_cols = ["LapTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
                "TeamStrength", "ExperienceScore", "NightFactor"]
X = merged_data[feature_cols].fillna(merged_data.mean())
y = merged_data["QualiPosition"]

# Train Random Forest with cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_split=3, random_state=42)
model.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"\n📊 Cross-Validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# Predict 2025 qualifying
X_2025 = fp3_2025[feature_cols]
predicted_positions = model.predict(X_2025)
fp3_2025["PredictedPosition"] = predicted_positions

# Sort and format results
fp3_2025 = fp3_2025.sort_values(by="PredictedPosition")
fp3_2025["PredictedPosition"] = fp3_2025["PredictedPosition"].round().astype(int)

print("\n🏁 Predicted 2025 Bahrain GP Qualifying Results 🏁\n")
for idx, row in fp3_2025.iterrows():
    print(f"P{row['PredictedPosition']:2d}: {row['Driver']} - {row['Team']} (FP3: {row['FP3Time (s)']:.3f}s)")

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n🎯 Test MAE: {mae:.2f} positions")

# Feature importance visualization
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n📈 Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
plt.xlabel('Importance Score')
plt.title('Feature Importance in Qualifying Position Prediction')
plt.tight_layout()
plt.savefig('feature_importance_bahrain.png')
print("\n✅ Feature importance chart saved as 'feature_importance_bahrain.png'")
