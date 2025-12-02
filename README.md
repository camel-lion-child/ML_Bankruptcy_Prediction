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

1   ROA(C) before interest and depreciation before interest  6734 non-null   float64
 2   ROA(A) before interest and  after tax                    6734 non-null   float64
 3   ROA(B) before interest and depreciation after tax        6734 non-null   float64
 4   Operating Gross Margin                                   6734 non-null   float64
 5   Realized Sales Gross Margin                              6734 non-null   float64
 6   Operating Profit Rate                                    6734 non-null   float64
 7   Pre-tax net Interest Rate                                6734 non-null   float64
 8   After-tax net Interest Rate                              6734 non-null   float64
 9   Non-industry income and expenditure/revenue              6734 non-null   float64
 10  Continuous interest rate (after tax)                     6734 non-null   float64
 11  Operating Expense Rate                                   6734 non-null   float64
 12  Research and development expense rate                    6734 non-null   float64
 13  Cash flow rate                                           6734 non-null   float64
 14  Interest-bearing debt interest rate                      6734 non-null   float64
 15  Tax rate (A)                                             6734 non-null   float64
 16  Net Value Per Share (B)                                  6734 non-null   float64
 17  Net Value Per Share (A)                                  6734 non-null   float64
 18  Net Value Per Share (C)                                  6734 non-null   float64
 19  Persistent EPS in the Last Four Seasons                  6734 non-null   float64
 20  Cash Flow Per Share                                      6734 non-null   float64
 21  Revenue Per Share (Yuan ¥)                               6734 non-null   float64
 22  Operating Profit Per Share (Yuan ¥)                      6734 non-null   float64
 23  Per Share Net profit before tax (Yuan ¥)                 6734 non-null   float64
 24  Realized Sales Gross Profit Growth Rate                  6734 non-null   float64
 25  Operating Profit Growth Rate                             6734 non-null   float64
 26  After-tax Net Profit Growth Rate                         6734 non-null   float64
 27  Regular Net Profit Growth Rate                           6734 non-null   float64
 28  Continuous Net Profit Growth Rate                        6734 non-null   float64
 29  Total Asset Growth Rate                                  6734 non-null   float64
 30  Net Value Growth Rate                                    6734 non-null   float64
 31  Total Asset Return Growth Rate Ratio                     6734 non-null   float64
 32  Cash Reinvestment                                        6734 non-null   float64
 33  Current Ratio                                            6734 non-null   float64
 34  Quick Ratio                                              6734 non-null   float64
 35  Interest Expense Ratio                                   6734 non-null   float64
 36  Total debt/Total net worth                               6734 non-null   float64
 37  Debt ratio                                               6734 non-null   float64
 38  Net worth/Assets                                         6734 non-null   float64
 39  Long-term fund suitability ratio (A)                     6734 non-null   float64
 40  Borrowing dependency                                     6734 non-null   float64
 41  Contingent liabilities/Net worth                         6734 non-null   float64
 42  Operating profit/Paid-in capital                         6734 non-null   float64
 43  Net profit before tax/Paid-in capital                    6734 non-null   float64
 44  Inventory and accounts receivable/Net value              6734 non-null   float64
 45  Total Asset Turnover                                     6734 non-null   float64
 46  Accounts Receivable Turnover                             6734 non-null   float64
 47  Average Collection Days                                  6734 non-null   float64
 48  Inventory Turnover Rate (times)                          6734 non-null   float64
 49  Fixed Assets Turnover Frequency                          6734 non-null   float64
 50  Net Worth Turnover Rate (times)                          6734 non-null   float64
 51  Revenue per person                                       6734 non-null   float64
 52  Operating profit per person                              6734 non-null   float64
 53  Allocation rate per person                               6734 non-null   float64
 54  Working Capital to Total Assets                          6734 non-null   float64
 55  Quick Assets/Total Assets                                6734 non-null   float64
 56  Current Assets/Total Assets                              6734 non-null   float64
 57  Cash/Total Assets                                        6734 non-null   float64
 58  Quick Assets/Current Liability                           6734 non-null   float64
 59  Cash/Current Liability                                   6734 non-null   float64
 60  Current Liability to Assets                              6734 non-null   float64
 61  Operating Funds to Liability                             6734 non-null   float64
 62  Inventory/Working Capital                                6734 non-null   float64
 63  Inventory/Current Liability                              6734 non-null   float64
 64  Current Liabilities/Liability                            6734 non-null   float64
 65  Working Capital/Equity                                   6734 non-null   float64
 66  Current Liabilities/Equity                               6734 non-null   float64
 67  Long-term Liability to Current Assets                    6734 non-null   float64
 68  Retained Earnings to Total Assets                        6734 non-null   float64
 69  Total income/Total expense                               6734 non-null   float64
 70  Total expense/Assets                                     6734 non-null   float64
 71  Current Asset Turnover Rate                              6734 non-null   float64
 72  Quick Asset Turnover Rate                                6734 non-null   float64
 73  Working capitcal Turnover Rate                           6734 non-null   float64
 74  Cash Turnover Rate                                       6734 non-null   float64
 75  Cash Flow to Sales                                       6734 non-null   float64
 76  Fixed Assets to Assets                                   6734 non-null   float64
 77  Current Liability to Liability                           6734 non-null   float64
 78  Current Liability to Equity                              6734 non-null   float64
 79  Equity to Long-term Liability                            6734 non-null   float64
 80  Cash Flow to Total Assets                                6734 non-null   float64
 81  Cash Flow to Liability                                   6734 non-null   float64
 82  CFO to Assets                                            6734 non-null   float64
 83  Cash Flow to Equity                                      6734 non-null   float64
 84  Current Liability to Current Assets                      6734 non-null   float64
 85  Liability-Assets Flag                                    6734 non-null   int64  
 86  Net Income to Total Assets                               6734 non-null   float64
 87  Total assets to GNP price                                6734 non-null   float64
 88  No-credit Interval                                       6734 non-null   float64
 89  Gross Profit to Sales                                    6734 non-null   float64
 90  Net Income to Stockholder's Equity                       6734 non-null   float64
 91  Liability to Equity                                      6734 non-null   float64
 92  Degree of Financial Leverage (DFL)                       6734 non-null   float64
 93  Interest Coverage Ratio (Interest expense to EBIT)       6734 non-null   float64
 94  Net Income Flag                                          6734 non-null   int64  
 95  Equity to Liability                                      6734 non-null   float64

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
