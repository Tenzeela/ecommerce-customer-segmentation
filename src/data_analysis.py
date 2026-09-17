import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#create results directory 
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_excel("data/Online Retail.xlsx")

# BASIC DATASET INFORMATION

print("\n========== DATASET OVERVIEW ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# =========================
# STATISTICAL SUMMARY
# =========================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# UNIQUE VALUES

print("\n========== UNIQUE VALUES ==========")

print("\nUnique customers:")
print(df['CustomerID'].nunique())

print("\nUnique products:")
print(df['StockCode'].nunique())

print("\nUnique countries:")
print(df["Country"].nunique())

print("\nUnique number of Transactions:")
print(df["InvoiceNo"].nunique())

# CANCELLATIONS
print("\n========== CANCELLATIONS ==========")

cancellations = df["InvoiceNo"].astype(str).str.startswith("C").sum()
print("Cancelled TransactionsS:", cancellations.sum())

# BASIC DATA CLEANING

print("\n========== CLEANING CHECK ==========")

print("Rows before cleaning:", len(df))

df = df.drop_duplicates()

df = df.dropna(subset=["CustomerID"])

df = df[df["Quantity"] > 0]

df = df[df["UnitPrice"] > 0]

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

print("Rows after cleaning:", len(df))

# CREATE TOTAL PRICE
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\n========== TOTAL PRICE ==========")

print(df["TotalPrice"].describe())

# COUNTRY DISTRIBUTION

print("\n========== TOP COUNTRIES ==========")

print(df["Country"].value_counts().head(10))

# VISUALIZATIONS

plt.figure(figsize=(10, 6))

sns.histplot(
    df["TotalPrice"],
    bins=50,
    kde=True
)

plt.title("Distribution of Transaction Value")
plt.xlabel("Transaction Value")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("results/transaction_value_distribution.png")

plt.close()


plt.figure(figsize=(10, 6))

top_countries = df["Country"].value_counts().head(10)

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Number of Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig("results/top_countries.png")

plt.close()


print("\n========== DATA ANALYSIS COMPLETE ==========")
print("EDA results saved in the results/ folder.")



# CUSTOMER-LEVEL FEATURES
print("\n========== CUSTOMER FEATURES ==========")

# Set the reference date

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_features = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    AvgOrderValue=("TotalPrice", "mean"),
    UniqueProducts=("StockCode", "nunique"),
    TotalQuantity=("Quantity", "sum")
).reset_index()

print("\nCustomer-level dataset shape:")
print(customer_features.shape)

print("\nFirst 5 customer records:")
print(customer_features.head())

print("\nCustomer feature statistics:")
print(customer_features.describe())

# Save customer-level dataset
customer_features.to_csv(
    "data/customer_features.csv",
    index=False
)

print("\nCustomer features saved to data/customer_features.csv")


# Save customer-level dataset
customer_features.to_csv(
    "data/customer_features.csv",
    index=False
)

print("\nCustomer features saved to data/customer_features.csv")