Bankruptcy Prediction Using Machine Learning
Predicting company bankruptcy using financial ratios & ML classification models

**Overview

This project builds a machine learning model to predict whether a company is likely to go bankrupt based on financial ratios, solvency indicators, leverage metrics, and profitability trends.

The objective is to create an end-to-end ML pipeline including:

ETL & data cleaning

Feature engineering

Exploratory data analysis

Model training & evaluation

Business interpretation of results

This type of model is widely used in risk analysis, credit scoring, factoring, and financial due diligence.

**Dataset

A public bankruptcy dataset (commonly used in finance research), including variables such as:

ROA(C) before interest and depreciation before interest,
ROA(A) before interest and after tax,
ROA(B) before interest and depreciation after tax,
Operating Gross Margin,
Realized Sales Gross Margin,
Operating Profit Rate,
Pre-tax net Interest Rate,
After-tax net Interest Rate,
Non-industry income and expenditure/revenue,
Continuous interest rate (after tax),
Operating Expense Rate,
Research and development expense rate,
Cash flow rate,
Interest-bearing debt interest rate,
Tax rate (A),
Net Value Per Share (B),
Net Value Per Share (A),
Net Value Per Share (C),
Persistent EPS in the Last Four Seasons,
Cash Flow Per Share,
Revenue Per Share (Yuan ¥),
Operating Profit Per Share (Yuan ¥),
Per Share Net profit before tax (Yuan ¥),
Realized Sales Gross Profit Growth Rate,
Operating Profit Growth Rate,
After-tax Net Profit Growth Rate,
Regular Net Profit Growth Rate,
Continuous Net Profit Growth Rate,
Total Asset Growth Rate,
Net Value Growth Rate,
Total Asset Return Growth Rate Ratio,
Cash Reinvestment,
Current Ratio,
Quick Ratio,
Interest Expense Ratio,
Total debt/Total net worth,
Debt ratio,
Net worth/Assets,
Long-term fund suitability ratio (A),
Borrowing dependency,
Contingent liabilities/Net worth,
Operating profit/Paid-in capital,
Net profit before tax/Paid-in capital,
Inventory and accounts receivable/Net value,
Total Asset Turnover,
Accounts Receivable Turnover,
Average Collection Days,
Inventory Turnover Rate (times),
Fixed Assets Turnover Frequency,
Net Worth Turnover Rate (times),
Revenue per person,
Operating profit per person,
Allocation rate per person,
Working Capital to Total Assets,
Quick Assets/Total Assets,
Current Assets/Total Assets,
Cash/Total Assets,
Quick Assets/Current Liability,
Cash/Current Liability,
Current Liability to Assets,
Operating Funds to Liability,
Inventory/Working Capital,
Inventory/Current Liability,
Current Liabilities/Liability,
Working Capital/Equity,
Current Liabilities/Equity,
Long-term Liability to Current Assets,
Retained Earnings to Total Assets,
Total income/Total expense,
Total expense/Assets,
Current Asset Turnover Rate,
Quick Asset Turnover Rate,
Working capitcal Turnover Rate,
Cash Turnover Rate,
Cash Flow to Sales,
Fixed Assets to Assets,
Current Liability to Liability,
Current Liability to Equity,
Equity to Long-term Liability,
Cash Flow to Total Assets,
Cash Flow to Liability,
CFO to Assets,
Cash Flow to Equity,
Current Liability to Current Assets,
Liability-Assets Flag,
Net Income to Total Assets,
Total assets to GNP price,
No-credit Interval,
Gross Profit to Sales,
Net Income to Stockholder's Equity,
Liability to Equity,
Degree of Financial Leverage (DFL),
Interest Coverage Ratio (Interest expense to EBIT),
Net Income Flag,
Equity to Liability.

**ETL Pipeline

Extract raw CSV data

Clean missing values, outliers, inconsistent financial codes

Transform ratios into standardized features

Load processed dataset for ML

Includes:

Scaling (MinMax / StandardScaler)

Winsorization

Feature normalization

**Machine Learning Models

Multiple classification models were tested:

Baseline Models

Logistic Regression

Decision Tree

Random Forest

Advanced Models

Gradient Boosting

XGBoost / LightGBM (optional)

Metrics Evaluated

Accuracy

Precision

Recall (bankruptcy recall = most important)

F1-score

ROC–AUC

Focus is placed on recall for bankrupt companies, since false negatives are extremely costly.

**Purpose

This project demonstrates:

End-to-end ML workflow

Financial modeling applied to company distress

Predictive analytics for credit risk & bankruptcy

Production-style organization
