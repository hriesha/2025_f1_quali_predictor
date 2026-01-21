import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Australian GP practice and qualifying sessions
session_practice_2024 = fastf1.get_session(2024, "Australia", "FP3")
session_practice_2024.load()

session_quali_2024 = fastf1.get_session(2024, "Australia", "Q")
session_quali_2024.load()

# Extract practice times
practice_laps = session_practice_2024.laps[["Driver", "LapTime"]].copy()
practice_laps.dropna(subset=["LapTime"], inplace=True)
practice_laps["PracticeTime (s)"] = practice_laps["LapTime"].dt.total_seconds()

# Average practice times per driver
avg_practice_times = practice_laps.groupby("Driver")["PracticeTime (s)"].mean().reset_index()

# Extract qualifying results for 2024
quali_results = session_quali_2024.results[["Abbreviation", "Position"]].copy()
quali_results.columns = ["Driver", "QualiPosition"]

# 2025 FP3 simulated data (in real scenario, this would be actual FP3 times)
fp3_2025 = pd.DataFrame({
    "Driver": ["NOR", "PIA", "VER", "RUS", "TSU", "ALB", "LEC", "HAM", "GAS", "SAI", "ALO", "STR"],
    "FP3Time (s)": [80.5, 80.7, 80.9, 81.1, 81.3, 81.5, 80.6, 81.2, 81.4, 81.0, 81.6, 81.8]
})

# Team form based on 2025 constructor standings
team_form = {
    "McLaren": 0.95, "Red Bull": 0.92, "Mercedes": 0.90, "Ferrari": 0.88,
    "Alpine": 0.85, "Aston Martin": 0.83, "Williams": 0.80, "Racing Bulls": 0.78
}

driver_to_team = {
    "NOR": "McLaren", "PIA": "McLaren", "VER": "Red Bull", "RUS": "Mercedes",
    "HAM": "Mercedes", "LEC": "Ferrari", "SAI": "Ferrari", "ALO": "Aston Martin",
    "STR": "Aston Martin", "GAS": "Alpine", "TSU": "Racing Bulls", "ALB": "Williams"
}

fp3_2025["Team"] = fp3_2025["Driver"].map(driver_to_team)
fp3_2025["TeamForm"] = fp3_2025["Team"].map(team_form)

# Circuit characteristic factor (some drivers perform better at specific circuits)
circuit_factor = {
    "NOR": 1.02, "PIA": 0.98, "VER": 1.01, "RUS": 0.99, "TSU": 0.97,
    "ALB": 1.00, "LEC": 0.98, "HAM": 1.01, "GAS": 0.96, "SAI": 0.99,
    "ALO": 1.00, "STR": 0.95
}

fp3_2025["CircuitFactor"] = fp3_2025["Driver"].map(circuit_factor)

# Merge with 2024 data
merged_data = avg_practice_times.merge(quali_results, on="Driver", how="inner")

# Create training features
merged_data["TeamForm"] = merged_data["Driver"].map(driver_to_team).map(team_form)
merged_data["CircuitFactor"] = merged_data["Driver"].map(circuit_factor)

X = merged_data[["PracticeTime (s)", "TeamForm", "CircuitFactor"]].fillna(merged_data.mean())
y = merged_data["QualiPosition"]

if X.shape[0] < 5:
    raise ValueError("Insufficient training data!")

# Train Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict 2025 qualifying positions
X_2025 = fp3_2025[["FP3Time (s)", "TeamForm", "CircuitFactor"]]
predicted_positions = model.predict(X_2025)
fp3_2025["PredictedPosition"] = predicted_positions

# Sort by predicted position
fp3_2025 = fp3_2025.sort_values(by="PredictedPosition")
fp3_2025["PredictedPosition"] = fp3_2025["PredictedPosition"].round().astype(int)

# Display predictions
print("\n🏁 Predicted 2025 Australian GP Qualifying Results 🏁\n")
for idx, row in fp3_2025.iterrows():
    print(f"P{row['PredictedPosition']}: {row['Driver']} ({row['Team']})")

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n📊 Model MAE: {mae:.2f} positions")
