import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from getpass import getpass

password = getpass("Enter PostgreSQL password: ")

db_url = URL.create(
    "postgresql+psycopg",
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

print(df.head())
print("Total transactions:", len(df))
print("Dataset shape:", df.shape)

print("\nColumn information:")
df.info()

print("\nStatistical summary:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate transactions:")
print(df.duplicated().sum())

print("\nDuplicate transaction IDs:")
print(df["transaction_id"].duplicated().sum())
counts = df["is_fraud"].value_counts().sort_index()

plt.bar(["Genuine", "Fraud"], counts.values)
plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.show()
plt.figure(figsize=(8, 5))

plt.hist(df["amount"], bins=20, edgecolor="black")

plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount ($)")
plt.ylabel("Number of Transactions")

plt.show()
category_counts = df["category"].value_counts()

plt.figure(figsize=(10, 5))
plt.bar(category_counts.index, category_counts.values)

plt.title("Transactions by Category")
plt.xlabel("Category")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))

plt.boxplot([
    df[df["is_fraud"] == 0]["amount"],
    df[df["is_fraud"] == 1]["amount"]
], tick_labels=["Genuine", "Fraud"])

plt.title("Transaction Amount: Genuine vs Fraud")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Amount ($)")

plt.show()
fraud_rate_by_hour = df.groupby("hour")["is_fraud"].mean() * 100

print("\nFraud Rate by Hour (%):")
print(fraud_rate_by_hour.round(2))
plt.figure(figsize=(10, 5))

plt.bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values)

plt.title("Fraud Rate by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(24))

plt.tight_layout()
plt.show()
