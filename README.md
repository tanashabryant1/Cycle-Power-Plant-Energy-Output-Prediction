# Cycle-Power-Plant-Energy-Output-Prediction
Cycle Power Plant Energy Output Predictions

# Combined Cycle Power Plant Energy Output Prediction

## Project Overview

# Cycle Power Plant Energy Output Prediction

The goal of the project was to apply the full machine learning modeling process, including:

- Defining the machine learning problem
- Selecting features and a target variable
- Splitting the dataset into training and test sets
- Using cross-validation
- Comparing multiple regression models
- Selecting the best-performing model
- Evaluating the final model on unseen test data

---

## Business Problem

Combined Cycle Power Plants use a combination of gas turbines, steam turbines, and heat recovery steam generators to produce electricity.

Environmental conditions can affect how efficiently a plant operates. This project uses ambient environmental measurements to predict the plant’s net hourly electrical energy output.

The prediction target is:

**PE — Net Electrical Energy Output in megawatts (MW)**

---

## Dataset

The dataset contains **9,568 hourly observations** collected from sensors at a Combined Cycle Power Plant.

### Features

| Variable | Description         |
| -------- | ------------------- |
| AT       | Ambient Temperature |
| V        | Exhaust Vacuum      |
| AP       | Ambient Pressure    |
| RH       | Relative Humidity   |

### Target

| Variable | Description                        |
| -------- | ---------------------------------- |
| PE       | Net Electrical Energy Output in MW |

The dataset contained no missing values, so no imputation was required before modeling.

---

## Machine Learning Approach

Because the target variable is continuous and numeric, this project uses a **supervised regression approach**.

Two regression models were compared:

1. **Multiple Linear Regression**
2. **Random Forest Regression**

The purpose of comparing these models was to determine whether a more flexible nonlinear algorithm could outperform a traditional linear baseline.

---

## Train/Test Split

The data was divided into:

- **80% Training Data**
- **20% Test Data**

The test set was held out during model selection and used only for final evaluation.

A `random_state` of `42` was used to make the results reproducible.

---

## Validation Strategy

A **5-fold cross-validation** strategy was used on the training dataset.

Cross-validation provides a more reliable estimate of model performance than relying on one fixed validation split.

The primary model-selection metric was:

### Root Mean Squared Error (RMSE)

RMSE was selected because it measures prediction error in the same unit as the target variable, which in this project is **megawatts**.

Lower RMSE indicates better predictive performance.

Additional metrics included:

- Mean Absolute Error (MAE)
- R-squared (R²)

---

## Model Results

| Model             | Cross-Validation RMSE | Test RMSE | Test MAE | Test R² |
| ----------------- | --------------------: | --------: | -------: | ------: |
| Linear Regression |                4.5738 |    4.5026 |   3.5959 |  0.9301 |
| Random Forest     |                3.4768 |    3.2691 |   2.3494 |  0.9632 |

The **Random Forest Regressor** achieved the lowest cross-validation RMSE and was selected as the final model.

---

## Final Model Performance

The final Random Forest model achieved approximately:

- **RMSE:** 3.27 MW
- **MAE:** 2.35 MW
- **R²:** 0.963

An R² score of approximately **0.963** indicates that the model explains about **96.3% of the variation in electrical power output** within the test dataset.

The MAE indicates that the model's predictions differ from the actual power output by approximately **2.35 MW on average**.

---

## Visualizations

The project includes several visualizations to support the analysis and model evaluation.

### 1. Distribution of Electrical Power Output

Shows the distribution of the target variable, PE.



### 2. Ambient Temperature vs. Power Output

Illustrates the relationship between ambient temperature and electrical output.



### 3. Correlation Matrix

Displays the relationships between the environmental variables and electrical power output.



### 4. Model Comparison

Compares Linear Regression and Random Forest using cross-validation RMSE.



### 5. Actual vs. Predicted Power Output

Shows how closely the Random Forest predictions match the actual test-set values.



### 6. Residual Analysis

Examines prediction errors from the Random Forest model.



### 7. Feature Importance

Shows which environmental variables had the greatest influence on the Random Forest model.



---

## Technologies Used

This project was completed in Python using:

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn

---

## Project Structure

```text
combined-cycle-power-plant-ml/
│
├── README.md
├── CCPP_data.csv
├── ccpp_model.py
├── ccpp_model_metrics.csv
│
├── visuals/
│   ├── 01_distribution_pe.png
│   ├── 02_at_vs_pe.png
│   ├── 03_correlation_matrix.png
│   ├── 04_model_comparison.png
│   ├── 05_actual_vs_predicted_rf.png
│   ├── 06_residual_plot_rf.png
│   └── 07_feature_importance_rf.png
│
└── presentation/
    └── CCPP_ML_Presentation.pptx
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate into the project folder

```bash
cd combined-cycle-power-plant-ml
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Run the Python script

```bash
python ccpp_model.py
```

The script will:

- Load and inspect the dataset
- Create exploratory visualizations
- Split the data into training and test sets
- Perform 5-fold cross-validation
- Compare Linear Regression and Random Forest
- Evaluate the final model
- Generate model performance metrics
- Save visualizations to the `visuals` folder

---

## Key Findings

The project demonstrated that Random Forest performed better than Linear Regression for predicting power plant energy output.

The main findings were:

- Random Forest produced a substantially lower RMSE.
- The final model explained approximately 96% of the variance in power output.
- Environmental variables provided strong predictive information.
- A nonlinear model captured relationships in the data more effectively than the linear baseline.
- Cross-validation helped ensure that the final model was selected based on consistent performance rather than a single validation split.

---

## Conclusion

The Random Forest Regressor was selected as the final machine learning model because it produced the best validation performance.

The model achieved strong predictive accuracy on previously unseen test data, demonstrating that ambient environmental conditions can be used effectively to estimate electrical energy output from a Combined Cycle Power Plant.

This project demonstrates the complete machine learning workflow from data exploration and model comparison through final evaluation and interpretation.

---

## Data Source

Pınar Tüfekci,
*Prediction of Full Load Electrical Power Output of a Base Load Operated Combined Cycle Power Plant Using Machine Learning Methods*,
International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126–140.

Heysem Kaya, Pınar Tüfekci, Sadık Fikret Gürgen,
*Local and Global Learning Methods for Predicting Power of a Combined Gas & Steam Turbine*,
Proceedings of ICETCEE 2012.

---

## Author

**Tanasha Bryant**

Interests and focus areas:

Machine Learning | Data Analytics | Business Intelligence | Product Management | Cybersecurity | Governance, Risk & Compliance
