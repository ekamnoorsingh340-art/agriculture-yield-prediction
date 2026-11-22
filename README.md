# 🌾 Indian Crop Yield Prediction Web App

An end-to-end Machine Learning web application that predicts expected agricultural crop yield (**tonnes per hectare**) and total harvest production across Indian states and cropping seasons using historical agronomic and meteorological data.

🔗 **Live Web Application:** [Launch App on Streamlit Cloud](https://agriculture-yield-prediction-zheolvwzb4p8qqkgwgnvtf.streamlit.app/)

---

## 📌 Project Overview
Accurate yield estimation supports agricultural planning, resource allocation, and supply-chain forecasting. This project trains a supervised regression pipeline on multi-state Indian crop production records and serves real-time predictions through an interactive Streamlit interface.

### Key Features
- **Multi-State & Multi-Crop Support:** Covers 30 Indian states/UTs, 30 major crops (including Wheat, Rice, Maize, Cotton, Sugarcane, and Pulses), and 6 cropping seasons (`Kharif`, `Rabi`, `Whole Year`, `Summer`, `Winter`, `Autumn`).
- **Zero-Leakage Feature Engineering:** Drops target-derived `Production` and temporal `Crop_Year` columns during preprocessing so predictions rely strictly on realistic pre-harvest inputs.
- **Real-Time Total Harvest Calculation:** Computes both unit yield (`tonnes/ha`) and estimated total production (`tonnes`) based on cultivated area.

---

## 📊 Dataset & Preprocessing
- **Source:** Indian Agricultural Crop Yield Dataset (`crop_yield.csv` — Kaggle)
- **Data Cleaning:**
  - Stripped trailing/leading whitespace across categorical string fields (`Crop`, `Season`, `State`).
  - Filtered out invalid records where `Area <= 0` or `Yield <= 0`.
  - Removed `Production` (to prevent target data leakage, since $\text{Yield} = \frac{\text{Production}}{\text{Area}}$) and `Crop_Year` (to prevent overfitting to historical year indices).
- **Input Features:**
  - **Categorical:** `Crop`, `Season`, `State` (encoded via `OneHotEncoder(handle_unknown="ignore")`)
  - **Numerical:** `Area` (Hectares), `Annual_Rainfall` (mm), `Fertilizer` (kg), `Pesticide` (kg)

---

## 🧠 Model Architecture & Performance
- **Algorithm:** `RandomForestRegressor` wrapped inside a `scikit-learn` `ColumnTransformer` + `Pipeline`
- **Train/Test Split:** 80% Training / 20% Testing (`random_state=42`)
- **Evaluation Metrics:**
  - **$R^2$ Score:** `0.9719`
  - **Mean Absolute Error (MAE):** `8.3711`
- **Model Serialization:** Exported via `joblib` (`yield_model.pkl`, ~1.57 MB) with pinned `scikit-learn==1.6.1` for deterministic production inference.

---

## 🗂️ Repository Structure
```text
agriculture-yield-prediction/
├── app.py               # Streamlit frontend and real-time inference script
├── yield_model.pkl      # Serialized Random Forest + OneHotEncoder pipeline
├── requirements.txt     # Pinned Python dependencies for Streamlit Cloud
├── LICENSE              # MIT License
└── README.md            # Project documentation
