
# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project demonstrates an end-to-end machine learning workflow
for credit card fraud classification using synthetic transaction data.

## Technologies Used
- Python
- PostgreSQL
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Random Forest
- Logistic Regression
- Streamlit

## Dataset
- 1,000 synthetic transactions
- 950 genuine transactions
- 50 fraud-labeled transactions

Important: Fraud labels were randomly generated, so the dataset
does not contain meaningful fraud patterns. This project is for
learning and demonstration purposes only.

## Machine Learning Workflow
1. Generate synthetic transaction data.
2. Store transactions in PostgreSQL.
3. Perform Exploratory Data Analysis (EDA).
4. Preprocess data and encode categories.
5. Split data into training and testing sets.
6. Train Logistic Regression and Random Forest models.
7. Evaluate using confusion matrices, precision, recall,
   F1-score, ROC-AUC, and Average Precision.
8. Experiment with classification threshold tuning.
9. Save the trained Random Forest model.
10. Build a Streamlit prediction application.

## Streamlit Application
Users can enter:
- Transaction amount
- Transaction hour
- Transaction category

The application displays a predicted class and model fraud score.

## Model Limitations
The models were trained on synthetic data with randomly assigned
fraud labels. Predictions are not suitable for real financial
decisions or production fraud detection.

## Author
Machine Learning Portfolio Project
