import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Enable caching
fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Monaco GP qualifying
try:
    session_2024 = fastf1.get_session(2024, "Monaco", "Q")
    session_2024.load()

    # Get 2024 qualifying results
    quali_2024 = session_2024.results[["Abbreviation", "Position", "Q3"]].copy()
    quali_2024.columns = ["Driver", "Position", "Q3Time"]
    quali_2024["Q3Time (s)"] = quali_2024["Q3Time"].dt.total_seconds()
    quali_2024.dropna(subset=["Q3Time (s)"], inplace=True)

except Exception as e:
    print(f"Note: Could not load 2024 Monaco data - {e}")
    # Create sample training data
    quali_2024 = pd.DataFrame({
        "Driver": ["LEC", "VER", "NOR", "SAI", "PIA", "RUS", "HAM", "ALO"],
        "Position": [1, 2, 3, 4, 5, 6, 7, 8],
        "Q3Time (s)": [72.3, 72.5, 72.6, 72.8, 73.0, 73.1, 73.2, 73.4]
    })

# Monaco specialization factor (street circuit experts)
monaco_specialists = {
    "LEC": 1.05, "VER": 1.03, "NOR": 1.00, "SAI": 0.99, "PIA": 0.96,
    "RUS": 0.98, "HAM": 1.04, "ALO": 1.02, "GAS": 0.97, "TSU": 0.94,
    "ALB": 0.98, "STR": 0.95, "HUL": 0.96, "OCO": 0.97
}

# 2025 Practice Data (Monaco - street circuit)
practice_2025 = pd.DataFrame({
    "Driver": ["LEC", "NOR", "VER", "PIA", "SAI", "RUS", "HAM", "ALO", "GAS", "TSU"],
    "BestPracticeLap (s)": [73.1, 73.3, 73.2, 73.5, 73.4, 73.6, 73.5, 73.8, 73.9, 74.1],
    "ConsistencyScore": [0.95, 0.92, 0.94, 0.90, 0.91, 0.88, 0.93, 0.89, 0.87, 0.85]
})

practice_2025["MonacoFactor"] = practice_2025["Driver"].map(monaco_specialists)

# Team current form (based on recent races)
team_form_score = {
    "McLaren": 0.98, "Ferrari": 0.96, "Red Bull": 0.94, "Mercedes": 0.92,
    "Alpine": 0.88, "Aston Martin": 0.87, "Racing Bulls": 0.85, "Williams": 0.83
}

driver_team_map = {
    "LEC": "Ferrari", "SAI": "Ferrari", "NOR": "McLaren", "PIA": "McLaren",
    "VER": "Red Bull", "RUS": "Mercedes", "HAM": "Mercedes", "ALO": "Aston Martin",
    "GAS": "Alpine", "TSU": "Racing Bulls", "ALB": "Williams", "STR": "Aston Martin",
    "HUL": "Haas", "OCO": "Alpine"
}

practice_2025["Team"] = practice_2025["Driver"].map(driver_team_map)
practice_2025["TeamForm"] = practice_2025["Team"].map(team_form_score)

# Prepare training data
quali_2024["MonacoFactor"] = quali_2024["Driver"].map(monaco_specialists)
quali_2024["TeamForm"] = quali_2024["Driver"].map(driver_team_map).map(team_form_score)
quali_2024["ConsistencyScore"] = 0.90  # Assumed baseline

X_train = quali_2024[["Q3Time (s)", "MonacoFactor", "TeamForm", "ConsistencyScore"]].fillna(0.9)
y_train = quali_2024["Position"]

# Train model
model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=77)
model.fit(X_train, y_train)

# Predict 2025 qualifying
X_2025 = practice_2025[["BestPracticeLap (s)", "MonacoFactor", "TeamForm", "ConsistencyScore"]]
X_2025.columns = ["Q3Time (s)", "MonacoFactor", "TeamForm", "ConsistencyScore"]

predictions = model.predict(X_2025)
practice_2025["PredictedPosition"] = predictions.round().astype(int)

# Sort and display
practice_2025 = practice_2025.sort_values("PredictedPosition")

print("\n🏎️ Predicted 2025 Monaco GP Qualifying Results 🏎️\n")
print("="*60)
for idx, row in practice_2025.iterrows():
    specialist_marker = "⭐" if row["MonacoFactor"] >= 1.02 else ""
    print(f"P{row['PredictedPosition']:2d}: {row['Driver']:3s} ({row['Team']:15s}) - {row['BestPracticeLap (s)']:.3f}s {specialist_marker}")

print("\n" + "="*60)
print(f"🎯 Monaco Circuit Specialists (⭐): High street circuit performance")
print(f"📊 Model trained on {len(X_train)} drivers from 2024 Monaco GP")
