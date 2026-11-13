from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from getpass import getpass
password = getpass("Enter PostgreSQL password: ")

db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password=password,
    host="localhost",
    port=5432,
    database="fraud_detection"
)

engine = create_engine(db_url)

with engine.connect() as connection:
    df = pd.read_sql(
        text("SELECT * FROM transactions"),
        connection
    )

print("Successfully loaded data from PostgreSQL!")
print("Total transactions:", len(df))
df_encoded = pd.get_dummies(
    df,
    columns=["category"],
    dtype=int
)

print("\nData after encoding:")
print(df_encoded.head())
X = df_encoded.drop(columns=["transaction_id", "is_fraud"])

y = df_encoded["is_fraud"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

password = getpass("Enter PostgreSQL password: ")
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# Logistic Regression predictions
lr_predictions = model.predict(X_test)

# Random Forest predictions
rf_predictions = rf_model.predict(X_test)

print("\nLogistic Regression Predictions:")
print(lr_predictions[:10])

print("\nRandom Forest Predictions:")
print(rf_predictions[:10])

print("\nActual Answers:")
print(y_test.iloc[:10].to_numpy())
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
from sklearn.metrics import classification_report

# Logistic Regression Evaluation
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions, zero_division=0))

# Random Forest Evaluation
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions, zero_division=0))
from sklearn.metrics import roc_auc_score, average_precision_score

# Step 1: Get fraud probability scores
lr_probabilities = model.predict_proba(X_test)[:, 1]
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# Step 2: Calculate ROC-AUC
lr_roc_auc = roc_auc_score(y_test, lr_probabilities)
rf_roc_auc = roc_auc_score(y_test, rf_probabilities)

# Step 3: Calculate Average Precision (PR-AUC summary)
lr_pr_auc = average_precision_score(y_test, lr_probabilities)
rf_pr_auc = average_precision_score(y_test, rf_probabilities)

# Step 4: Display results
print("\nLogistic Regression:")
print("ROC-AUC:", round(lr_roc_auc, 3))
print("Average Precision:", round(lr_pr_auc, 3))

print("\nRandom Forest:")
print("ROC-AUC:", round(rf_roc_auc, 3))
print("Average Precision:", round(rf_pr_auc, 3))
# Step 8: Try a lower fraud threshold
threshold = 0.30

rf_new_predictions = (rf_probabilities >= threshold).astype(int)

print("\nRandom Forest with Threshold 0.30:")
print(confusion_matrix(y_test, rf_new_predictions))

print(classification_report(
    y_test,
    rf_new_predictions,
    zero_division=0
))

# Step 9.1: Save the trained Random Forest model

import joblib
import os

# Create a folder to store the model
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(rf_model, "models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
