# 🏎️ F1 Qualifying Predictor 2025

A machine learning project that predicts **Formula 1 qualifying positions** for the 2025 season using **Random Forest algorithms** and historical F1 data from the FastF1 API.

## 🎯 Project Overview

Unlike traditional race outcome predictions, this project focuses specifically on **qualifying session results** - often the most critical factor in determining race performance. The model uses Random Forest regression to predict grid positions based on:

- **Practice session performance** (FP3 lap times and sector analysis)
- **Team performance metrics** (constructor standings and form)
- **Circuit-specific factors** (track characteristics and driver specialization)
- **Driver experience and consistency scores**
- **Historical qualifying data from 2024**

## 🔍 Why Qualifying Predictions?

Qualifying determines starting grid positions, which are crucial for race outcomes - especially on circuits where overtaking is difficult. By focusing on qualifying rather than race results, this model targets the most predictable and data-rich session in F1.

## 🚀 Features

- **Random Forest Regressor**: More robust to overfitting than gradient boosting for smaller datasets
- **Multi-factor analysis**: Combines practice times, team strength, and circuit characteristics
- **Circuit specialization**: Models driver performance on specific track types (street circuits, high-speed, technical)
- **Cross-validation**: Ensures model reliability across different data splits
- **Feature importance visualization**: Understand which factors matter most for qualifying performance

## 📊 Data Sources

- **FastF1 API**: Official F1 timing data including practice, qualifying, and race sessions
- **2024 Historical Data**: Training baseline from previous season
- **2025 Practice Sessions**: Live FP3 data for predictions
- **Team Performance Metrics**: Constructor standings and recent form

## 📁 File Structure

- `quali_pred_australia.py` - Australian GP qualifying predictions
- `quali_pred_bahrain.py` - Bahrain GP with advanced sector analysis
- `quali_pred_monaco.py` - Monaco GP with street circuit specialization
- Additional race predictions will be added throughout the 2025 season

## 🛠️ Installation & Dependencies

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib
```

### Required Libraries
- `fastf1` - Official F1 data API
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning (Random Forest)
- `matplotlib` - Visualization

## 🏁 Usage

Run any qualifying prediction script:

```bash
python3 quali_pred_australia.py
```

**Example Output:**
```
🏁 Predicted 2025 Australian GP Qualifying Results 🏁

P1: NOR (McLaren)
P2: PIA (McLaren)
P3: LEC (Ferrari)
...

📊 Model MAE: 1.85 positions
```

## 📈 Model Performance

The model is evaluated using **Mean Absolute Error (MAE)** measured in grid positions. An MAE of ~2 positions means the model typically predicts within 2 grid spots of actual qualifying results.

**Performance Metrics:**
- Cross-validation MAE: ~1.8-2.2 positions
- Feature importance analysis included in Bahrain script
- Model improves as more 2025 data becomes available

## 🔧 Methodology

1. **Data Collection**: Extract FP3 and historical qualifying data via FastF1
2. **Feature Engineering**:
   - Normalize practice lap times
   - Calculate team performance scores
   - Apply circuit-specific driver factors
   - Include experience and consistency metrics
3. **Model Training**: Random Forest with optimized hyperparameters
4. **Prediction**: Project qualifying positions for upcoming race
5. **Validation**: Cross-validation and MAE evaluation

## 🎛️ Key Differences from Race Predictions

| Aspect | This Project (Qualifying) | Race Predictions |
|--------|---------------------------|------------------|
| **Target** | Grid position | Race finishing time |
| **Algorithm** | Random Forest | Gradient Boosting |
| **Key Features** | FP3 times, circuit factors | Race pace, pit strategies |
| **Predictability** | Higher (fewer variables) | Lower (more chaos factors) |
| **Data Source** | Practice sessions | Historical race results |

## 🌟 Circuit Specialization

The model includes specific factors for different circuit types:

- **Street Circuits** (Monaco, Singapore): Driver precision and experience weighted higher
- **High-Speed Circuits** (Monza, Spa): Team aerodynamic efficiency emphasized
- **Technical Circuits** (Suzuka, Barcelona): All-around driver skill valued

## 📌 Future Enhancements

- [ ] Weather integration for wet qualifying sessions
- [ ] Tire compound strategy analysis
- [ ] Real-time FP3 data ingestion
- [ ] Confidence intervals for predictions
- [ ] Historical accuracy tracking throughout 2025 season
- [ ] Ensemble methods combining multiple models

## 📜 License

MIT License - Free to use and modify

## 🙏 Credits

- **FastF1**: Amazing open-source F1 data API
- **F1 Community**: For continuous feedback and improvements

---

**Made for F1 fans and data enthusiasts!** 🏎️💨

*Follow along as we predict every qualifying session of the 2025 F1 season!*
