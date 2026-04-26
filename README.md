# Energy Demand Predictor for Commercial Buildings

A supervised machine learning project to predict hourly energy consumption of commercial buildings using regression models, built on the ASHRAE Great Energy Predictor III dataset.

📊 [View Analysis Notebook](notebooks/01_energy_demand_analysis.ipynb) · 🚀 [Live Demo on Hugging Face](https://huggingface.co/spaces/Devina707/M2-Energy-Demand-Predictor)

---

## Problem Background

Buildings account for approximately 40% of global energy consumption, making them a significant contributor to greenhouse gas (GHG) emissions. As sustainability initiatives become a top global priority, the ability to accurately predict a building's energy demand is increasingly critical — enabling more efficient resource allocation and meaningful environmental and economic improvements.

This project addresses that need by building a machine learning model capable of predicting hourly energy usage for buildings in a specific site, based on structural and weather-related features.

---

## Project Output

- **ML Model**: A regression model trained to predict hourly energy demand for buildings in Site 1
- **EDA Report**: Comprehensive exploratory data analysis of the ASHRAE dataset
- **Deployed App**: Interactive prediction interface hosted on Hugging Face Spaces

**Best Model**: Histogram-based Gradient Boosting Regressor

| Metric | Score |
|--------|-------|
| MAE |27.8839|
| RMSE |50.8187|
| MSE |2582.5352|
| R² |0.9178|


---

## Repository Structure

```
├── data/
│   └── raw/                        # Original dataset
├── notebooks/
│   ├── 01_energy_demand_analysis.ipynb   # Full analysis: EDA, feature engineering, modeling, conclusions
│   └── 02_inference.ipynb               # Model inference and prediction testing
├── models/                         # Saved model artifacts (see Models section)
└── README.md
└── requirements.txt
```

---

## Data

| Detail | Info |
|--------|------|
| **Source** | [ASHRAE Great Energy Predictor III — Kaggle](https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset/data?select=building_metadata.csv) |
| **Coverage** | 1,400+ buildings across 16 sites (UK, US, Europe) — 2016 to 2018 |
| **Size** | 20M+ rows, 17 columns |
| **Features** | 3 categorical, 14 numerical |

**Columns with missing values:** `year_built`, `air_temperature`, `floors`, `cloud_coverage`, `dew_temperature`, `precip_depth_1_hr`, `sea_level_pressure`, `wind_direction`. Missing values are handled in the feature engineering stage of the analysis notebook.

---

## Method

This is a supervised regression task. Five models were trained and evaluated before selecting the best performer for deployment:

- K-Nearest Neighbors (KNN)
- Support Vector Regressor (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- Histogram-based Gradient Boosting Regressor *(best model)*

**Evaluation metrics:** MAE, MSE, RMSE, R²
**Hyperparameter tuning:** GridSearchCV with KFold cross-validation (CV RMSE, CV RMSE std)

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/energy-demand-predictor.git
cd energy-demand-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the analysis notebook**
```
notebooks/01_energy_demand_analysis.ipynb
```

---

## Models

Pre-trained model artifacts are available for download:

| File | Description |
|------|-------------|
| `pipelines_inference.pkl` | Trained Histogram-based Gradient Boosting model pipeline |
| `model_inf.pkl` | Best model Chosen: Histogram-based Gradient Boosting model pipeline |
| `preprocessing_function.pkl` | Final features for modeling after feature selection |
| `encoder.pkl` | Fitted categorical encoder |
| `scaler.pkl` | Fitted scaler for numerical feature normalization |
| `cat_col.txt` | Saved categorical column lists |
| `num_col.txt` | Saved numerical column lists |


Place downloaded files in the `models/` directory before running `02_inference.ipynb`.

---

## Stacks

| Category | Libraries |
|----------|-----------|
| **Data & Analysis** | pandas, numpy, scipy (`spearmanr`, `pearsonr`, `f_oneway`), statsmodels |
| **Visualization** | matplotlib, seaborn, shap |
| **Modeling** | scikit-learn (KNN, SVR, Decision Tree, Random Forest, HistGradientBoosting) |
| **Utilities** | joblib, dill, pickle, json |

---

## References

- **Dataset**: [ASHRAE Great Energy Predictor III — Kaggle](https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset/data?select=building_metadata.csv)
- **Deployed App**: [Hugging Face Spaces](https://huggingface.co/spaces/Devina707/M2-Energy-Demand-Predictor)
- **Topic Reference**: [ScienceDirect — Building Energy Consumption](https://www.sciencedirect.com/science/article/pii/S2352484723013975)

**Model Documentation:**
- [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

---

*Built by Devina Agustina*
