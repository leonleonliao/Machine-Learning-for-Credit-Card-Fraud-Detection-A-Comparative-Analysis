# Machine Learning for Credit Card Fraud Detection

This project develops and evaluates machine learning models to detect fraudulent credit card transactions.

## Problem Description

This is a supervised learning, binary classification task aimed at distinguishing between legitimate and fraudulent transactions. The primary goal is to maximize the detection of fraud (Recall) while managing false positives.

## Data Source

The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle.
- **URL:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Citation:** Dal Pozzolo, A., Caelen, O., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification.

To run the notebook, please download the `creditcard.csv` file from the link above and place it in the same directory as the notebook.

## Methodology

1.  **Exploratory Data Analysis (EDA):** Identified extreme class imbalance (0.17% fraud).
2.  **Preprocessing:** Scaled `Time` and `Amount` features using `StandardScaler`.
3.  **Handling Imbalance:** Applied **Random Undersampling** on the training set to create a balanced dataset for model training.
4.  **Modeling:** Compared two models:
    - Logistic Regression
    - Random Forest
5.  **Evaluation:** Used metrics suitable for imbalanced data, including Recall, Confusion Matrix, and the Precision-Recall Curve (AUPRC).

## Results

The **Random Forest** model, trained on the undersampled data, was the best performer.
- **Fraud Recall:** 92%
- **Area Under the PR Curve (AUPRC):** 0.814

This indicates a strong ability to identify most fraudulent transactions while maintaining a good balance of precision.

## How to Run

1.  Ensure you have Python and Jupyter Notebook installed.
2.  Install the necessary libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Download `creditcard.csv` from the data source link and place it in the repository's root folder.
4.  Open and run the `Project-IntroML.ipynb` notebook.

