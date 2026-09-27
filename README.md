# Zillow Home Value Prediction

An **automated valuation model (AVM)** estimates the value of a property using statistical and machine learning models instead of a manual appraisal. Zillow's AVM is the "Zestimate".

This project predicts the error of the Zestimate, defined as:

logerror = log(Zestimate) - log(SalePrice)

## Scope

- **Data:** 2016 property features and 2016 transactions from the Kaggle Zillow Prize competition (https://www.kaggle.com/competitions/zillow-prize-1).
- **Preparation:** drop features with more than 95% missing values, remove duplicate or unmapped columns, fill missing values, and average the log error for properties sold more than once.
- **Models:** Linear Regression (baseline), Gradient Boosted Trees, XGBoost, a simple neural network, a neural network with regularisation, and CatBoost.
- **Evaluation:** MSE, RMSE and R-squared on a holdout set.
- **Explainability:** feature importance and SHAP values.

## Running the notebook

The data is not included. Download the csv files from Kaggle and place them in the same folder as `Zillow_AVM.ipynb`.
