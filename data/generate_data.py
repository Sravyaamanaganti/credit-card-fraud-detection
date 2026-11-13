import random
import pandas as pd

random.seed(42)

categories = [
    "Grocery",
    "Electronics",
    "Shopping",
    "Rent",
    "Movie",
    "Travel",
    "Jewellery"
]

transactions = []

for i in range(1000):

    transaction_id = 1001 + i
    amount = round(random.uniform(5, 3000), 2)
    hour = random.randint(0, 23)
    category = random.choice(categories)

    is_fraud = random.choices(
        [0, 1],
        weights=[95, 5]
    )[0]

    transactions.append([
        transaction_id,
        amount,
        hour,
        category,
        is_fraud
    ])

df = pd.DataFrame(
    transactions,
    columns=[
        "transaction_id",
        "amount",
        "hour",
        "category",
        "is_fraud"
    ]
)

print(df.head())
print("Total transactions:", len(df))
from sqlalchemy import create_engine
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
    print("Successfully connected to PostgreSQL!")
df.to_sql(
    "transactions",
    engine,
    if_exists="replace",
    index=False
)

print("Successfully stored 1000 transactions in PostgreSQL!")
