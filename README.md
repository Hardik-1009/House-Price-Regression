# 🏠 House Price Prediction using Machine Learning

This project builds a statistically grounded and production-ready machine learning system to predict residential house sale prices using structured property data from the Kaggle *House Prices — Advanced Regression Techniques* dataset.

The workflow explicitly includes:
- Statistical feature analysis,
- Model training and tuning,
- **Model diagnostics and validation**,
- **Uncertainty estimation via prediction intervals**,
- And deployment readiness.

---

## 📌 Problem Statement

Given detailed information about a house (size, location, quality, age, etc.), predict its final sale price as accurately and reliably as possible, while also quantifying prediction uncertainty.

This is a supervised regression problem with:
- A continuous target variable (`SalePrice`)
- Mixed feature types (numeric, categorical, ordinal)
- Non-linear relationships
- Heteroscedastic noise

---

## 📁 Dataset

**Source:** Kaggle — House Prices: Advanced Regression Techniques  
**Observations:** 1,460 training samples  
**Features:** 79 predictors  
**Target:** `SalePrice`

Features describe:
- Physical attributes (area, rooms, age),
- Location and neighborhood,
- Quality and condition ratings,
- Garage, basement, and exterior details.

---

## 🧭 End-to-End Workflow

1. Data ingestion and cleaning  
2. Missing value imputation and encoding  
3. Statistical feature analysis  
4. Model training and hyperparameter tuning  
5. **Model diagnostics (residuals, bias, heteroscedasticity)**  
6. **Prediction uncertainty estimation (intervals + calibration)**  
7. Final model selection  
8. Deployment preparation  

---

## 🧹 Data Preprocessing

### Missing Value Handling
- Structural absence → `"None"` or `0` (e.g., no garage, no fireplace)
- Continuous variables → median imputation
- Categorical variables → most frequent category

### Feature Encoding
- Ordinal features encoded as: `None < Po < Fa < TA < Gd < Ex`
- Nominal categorical features one-hot encoded
- Numeric features scaled where appropriate

All preprocessing was implemented inside Scikit-learn pipelines to prevent data leakage.

---

## 📊 Statistical Feature Analysis

| Feature Type | Test Used |
|--------------|-----------|
Numeric → Target | Spearman Rank Correlation |
Categorical → Target | Kruskal–Wallis H Test |
Ordinal → Target | Kruskal–Wallis H Test |

Only statistically significant features were retained for modeling.

---

## 🤖 Modeling

Models evaluated:
- Linear, Ridge, Lasso
- Decision Tree, Random Forest
- Gradient Boosting
- Support Vector Regression
- XGBoost

Cross-validated hyperparameter tuning was applied to all major models.

---

## 🏆 Final Model

**Model:** XGBoost Regressor  
**Test R²:** ≈ 0.923  
**Test RMSE:** ≈ 24,348  
**Test MAE:** ≈ 15,395  
**Test MAPE:** ≈ 9.35%

### Best Hyperparameters

```yaml
learning_rate: 0.03
max_depth: 3
n_estimators: 800
subsample: 0.7
colsample_bytree: 0.7
reg_alpha: 1
reg_lambda: 1
