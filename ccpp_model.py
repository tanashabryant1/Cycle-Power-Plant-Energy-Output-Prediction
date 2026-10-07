"""
Combined Cycle Power Plant Machine Learning Project

Goal:
Predict net hourly electrical energy output (PE) using environmental
sensor readings from a Combined Cycle Power Plant.

Models compared:
1. Linear Regression
2. Random Forest Regression

Evaluation:
- 5-fold cross-validation
- RMSE
- MAE
- R-squared
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

DATA_FILE = "CCPP_data.csv"
VISUALS_DIR = "visuals"
RANDOM_STATE = 42

os.makedirs(VISUALS_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\nDATASET PREVIEW")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())


# ---------------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# ---------------------------------------------------------

# Features
X = df[["AT", "V", "AP", "RH"]]

# Target
y = df["PE"]


# ---------------------------------------------------------
# 4. EXPLORATORY VISUALIZATIONS
# ---------------------------------------------------------

# Visual 1: Distribution of Power Output
plt.figure(figsize=(9, 6))
plt.hist(df["PE"], bins=30)
plt.title("Distribution of Electrical Power Output")
plt.xlabel("Power Output (MW)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "01_distribution_pe.png"),
    dpi=200
)
plt.close()


# Visual 2: Ambient Temperature vs Power Output
plt.figure(figsize=(9, 6))
plt.scatter(df["AT"], df["PE"], alpha=0.5)
plt.title("Ambient Temperature vs Power Output")
plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("Power Output (MW)")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "02_at_vs_pe.png"),
    dpi=200
)
plt.close()


# Visual 3: Correlation Matrix
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)
plt.colorbar()
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "03_correlation_matrix.png"),
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTRAINING SET SIZE")
print(X_train.shape)

print("\nTEST SET SIZE")
print(X_test.shape)


# ---------------------------------------------------------
# 6. CREATE MODELS
# ---------------------------------------------------------

linear_model = LinearRegression()

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ---------------------------------------------------------
# 7. 5-FOLD CROSS-VALIDATION
# ---------------------------------------------------------

linear_cv_scores = cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

rf_cv_scores = cross_val_score(
    random_forest_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

linear_cv_rmse = np.sqrt(-linear_cv_scores.mean())
rf_cv_rmse = np.sqrt(-rf_cv_scores.mean())

print("\nCROSS-VALIDATION RESULTS")
print(f"Linear Regression CV RMSE: {linear_cv_rmse:.4f}")
print(f"Random Forest CV RMSE:     {rf_cv_rmse:.4f}")


# ---------------------------------------------------------
# 8. MODEL COMPARISON VISUAL
# ---------------------------------------------------------

model_names = ["Linear Regression", "Random Forest"]
cv_rmse_scores = [linear_cv_rmse, rf_cv_rmse]

plt.figure(figsize=(8, 6))
plt.bar(model_names, cv_rmse_scores)
plt.title("Model Comparison Using Cross-Validation RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE (MW)")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "04_model_comparison.png"),
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 9. TRAIN BOTH MODELS
# ---------------------------------------------------------

linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)


# ---------------------------------------------------------
# 10. TEST SET PREDICTIONS
# ---------------------------------------------------------

linear_predictions = linear_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)


# ---------------------------------------------------------
# 11. EVALUATE MODELS
# ---------------------------------------------------------

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)
linear_mae = mean_absolute_error(
    y_test, linear_predictions
)
linear_r2 = r2_score(
    y_test, linear_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)
rf_mae = mean_absolute_error(
    y_test, rf_predictions
)
rf_r2 = r2_score(
    y_test, rf_predictions
)


results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "Cross Validation RMSE": [
        linear_cv_rmse,
        rf_cv_rmse
    ],
    "Test RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "Test MAE": [
        linear_mae,
        rf_mae
    ],
    "Test R-Squared": [
        linear_r2,
        rf_r2
    ]
})

print("\nMODEL COMPARISON")
print(results.round(4))

results.to_csv(
    "ccpp_model_metrics.csv",
    index=False
)


# ---------------------------------------------------------
# 12. ACTUAL VS PREDICTED
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.6
)

minimum_value = min(
    y_test.min(),
    rf_predictions.min()
)

maximum_value = max(
    y_test.max(),
    rf_predictions.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value]
)

plt.title("Random Forest: Actual vs Predicted Power Output")
plt.xlabel("Actual Power Output (MW)")
plt.ylabel("Predicted Power Output (MW)")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "05_actual_vs_predicted_rf.png"),
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 13. RESIDUAL ANALYSIS
# ---------------------------------------------------------

residuals = y_test - rf_predictions

plt.figure(figsize=(8, 6))
plt.scatter(
    rf_predictions,
    residuals,
    alpha=0.6
)
plt.axhline(y=0)
plt.title("Random Forest Residual Plot")
plt.xlabel("Predicted Power Output (MW)")
plt.ylabel("Residual Error (MW)")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "06_residual_plot_rf.png"),
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 14. FEATURE IMPORTANCE
# ---------------------------------------------------------

feature_importance = pd.Series(
    random_forest_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(feature_importance)

plt.figure(figsize=(8, 6))
plt.bar(
    feature_importance.index,
    feature_importance.values
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Environmental Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(VISUALS_DIR, "07_feature_importance_rf.png"),
    dpi=200
)
plt.close()


# ---------------------------------------------------------
# 15. SAMPLE PREDICTIONS
# ---------------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual PE": y_test.values,
    "Predicted PE": rf_predictions,
    "Prediction Error": y_test.values - rf_predictions
})

print("\nSAMPLE RANDOM FOREST PREDICTIONS")
print(prediction_results.head(10).round(2))


# ---------------------------------------------------------
# 16. FINAL MODEL SELECTION
# ---------------------------------------------------------

if rf_cv_rmse < linear_cv_rmse:
    final_model_name = "Random Forest Regression"
else:
    final_model_name = "Linear Regression"

print("\nFINAL MODEL")
print(final_model_name)


# ---------------------------------------------------------
# 17. FINAL RESULTS
# ---------------------------------------------------------

print("\nFINAL RANDOM FOREST PERFORMANCE")
print(f"Test RMSE: {rf_rmse:.2f} MW")
print(f"Test MAE:  {rf_mae:.2f} MW")
print(f"Test R²:   {rf_r2:.3f}")

print("\nVisuals saved in the 'visuals' folder.")
print("Model metrics saved as 'ccpp_model_metrics.csv'.")
