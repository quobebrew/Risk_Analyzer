# Credit Risk Analyzer

## Project Overview

This project focuses on developing a machine learning model to assess credit risk for loan applicants. Credit risk is a crucial concern for financial institutions, as even approved loans carry the risk of default, impacting the institution's bottom line. By predicting the likelihood of default or non-performance for active loans, financial firms can proactively manage their risk exposure.

## Dataset and Problem Statement

The dataset used for this analysis comprises 10,000 records from Lending Club, a peer-to-peer lending platform. The target variable is the "Loan Status" column, which includes categories such as Paid Off, Current, In Grace Period, Late, and Charged Off. The goal is to predict the risk category of each loan, with '0' denoting healthy loans and '1' indicating loans at risk of default.

Dataset Link: [Lending Club Data - Loans Full Schema Dataset](https://www.openintro.org/data/index.php?data=loans_full_schema)

## Data Preprocessing and Cleansing

The dataset undergoes thorough preprocessing and cleansing:
- Null values are handled appropriately, and categorical variables are encoded.
- The target variable is reshaped to create a binary classification of healthy and risky loans.
- Numerical features are scaled using ScikitLearn's standard scaler.
- The dataset is rebalanced to address the heavily imbalanced nature of the risk category variable.

## Data Analysis and Visualization

- Summary statistics for numeric columns are computed.
- Horizontal bar plots visualize the frequency of each category for object columns, providing insights into the distribution of categorical variables.

## Machine Learning Models

Several machine learning models are trained and evaluated:
- Support Vector Machine (SVM)
- Logistic Regression
- ADA Boost
- Random Forest
- Neural Networks

Each model is assessed based on metrics such as precision, recall, F1-score, and balanced accuracy score.

## Model Validation and Comparison

K-Fold Cross Validation is employed to validate and compare the performance of the models. The aim is to select the model with the highest predictive accuracy and robustness for credit risk assessment.

## Key Findings

- The AdaBoost Classifier emerged as the best-performing model with a balanced accuracy score of 0.63, showcasing promising potential for accurately predicting credit risk.
- Data preprocessing techniques, including feature scaling and rebalancing, significantly improved model performance and generalization.
- Neural network models exhibited competitive performance but required more computational resources and longer training times compared to traditional machine learning algorithms.
- Interpretability of the models varied, with decision tree-based models such as Random Forest providing insight into feature importance for credit risk assessment.

## Conclusion

The project aims to provide financial institutions with a reliable tool for assessing credit risk, enabling them to make informed lending decisions and mitigate potential losses due to loan defaults. Through meticulous data preprocessing, model training, and validation, the project strives to deliver actionable insights that contribute to the stability and sustainability of financial operations.

The Results Can be seen in main.ipynb
