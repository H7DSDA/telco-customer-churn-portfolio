# Telco Customer Churn Portfolio

## 1. Project Overview

This repository contains a complete data science portfolio project
focused on customer churn prediction for a telecommunications company.\
The project follows a full data science workflow --- from data
preprocessing, exploratory data analysis (EDA), feature selection,
modeling, and interpretation, to dashboard visualization.

## 2. Objectives

-   Identify key factors influencing customer churn.\
-   Build an interpretable and accurate predictive model.\
-   Visualize insights and results in an interactive dashboard.

## 3. Dataset

The dataset contains demographic, service usage, and billing information
for telecom customers.\
Each record includes whether a customer has churned or not, making it
suitable for supervised classification tasks.

## 4. Project Workflow

1.  **Data Cleaning & Preprocessing**
    -   Handle missing values, duplicates, and categorical encoding.\
    -   Create new features for better model performance.
2.  **Exploratory Data Analysis (EDA)**
    -   Identify churn patterns using univariate and bivariate
        analysis.\
    -   Detect correlations and visual trends.
3.  **Modeling & Evaluation**
    -   Compare multiple algorithms (Logistic Regression, Random Forest,
        XGBoost).\
    -   Use SMOTE for handling class imbalance.\
    -   Evaluate models using Accuracy, Precision, Recall, F1-Score,
        MAE, MAPE, and RMSE.
4.  **Model Interpretation**
    -   Apply feature importance visualization and SHAP for
        explainability.
5.  **Dashboard Visualization**
    -   Developed using Streamlit for interactive model results.\
    -   Includes filtering, performance charts, and churn segmentation.

## 5. Repository Structure

    ├── 1_Telco Customer Churn
    │   ├── data_clean/            → Cleaned dataset files
    │   ├── models/                → Trained models (.joblib)
    │   ├── notebooks/             → EDA and modeling notebooks
    │   └── dashboard/             → Streamlit app scripts and requirements.txt
    ├── LICENSE
    └── README.md

## 6. Tools & Libraries

-   **Python**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib,
    Seaborn\
-   **Streamlit**: Interactive dashboard\
-   **Power BI**: Business insights visualization\
-   **Google Colab**: Model development environment

## 7. How to Run the Dashboard (Local)

``` bash
git clone https://github.com/H7DSDA/telco-customer-churn-portfolio.git
cd telco-customer-churn-portfolio/1_Telco Customer Churn/dashboard
pip install -r requirements.txt
streamlit run churn_dashboard_app_v2.py
```

## 8. Results & Insights

-   Customers with long tenure and bundled services show lower churn
    probability.\
-   High monthly charges and electronic payment methods correlate with
    higher churn risk.\
-   The XGBoost model provides the best accuracy and interpretability.

## 9. Author

**Hans Christian (H7DSDA)**\
Email: hans.dsda771@gmail.com\
Location: Bandung, Indonesia
