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
plt.figure(figsize=(10, 5))

plt.hist(df["hour"], bins=24, range=(0, 24), edgecolor="black")

plt.title("Transactions by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Transactions")
plt.xticks(range(0, 24, 2))

plt.show()
category_counts = df["category"].value_counts()

plt.figure(figsize=(10, 5))

plt.bar(category_counts.index, category_counts.values)

plt.title("Transactions by Category")
plt.xlabel("Transaction Category")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()