import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "synthetic_customer_churn_100k.csv"
OUTPUT = "output"
FIGURES = "figures"

def mkdir():
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def read_data(df):
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    print(df.duplicated().sum())
    print(df.isna().sum())
    print(df.describe())
    print(df.info())
    cols = df.select_dtypes(include="object").columns
    for col in cols:
        print(f"-------{col}-------")
        print(df[col].describe())

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    df = df[(df["age"] > 0) & (df["age"] < 100)]
    df = df[(df["monthlycharges"] > 0)]
    df = df[(df["totalcharges"] > 0)]
    return df

def analysis_data(df):
    top10_old_age = df.sort_values("age",ascending=False).head(10)
    print(top10_old_age[["customerid", "age", "churn"]])
    top10_high_monthlycharges = df.sort_values("monthlycharges",ascending=False).head(10)
    print(top10_high_monthlycharges[["customerid", "monthlycharges", "churn"]])
    top10_high_totalcharges = df.sort_values("totalcharges",ascending=False).head(10)
    print(top10_high_totalcharges[["customerid", "totalcharges", "churn"]])
    avg_monthlycharges_by_churn = df.groupby("churn")["monthlycharges"].mean()
    print(avg_monthlycharges_by_churn)
    age_tenure = df[["age", "tenure"]].agg(["mean", "median", "min", "max"])
    print(age_tenure)
    churn_age = df.groupby("churn")["age"].agg(["mean", "median", "min", "max"])
    print(churn_age)

    churn_summary = df.groupby("churn").agg(
    avg_age=("age", "mean"),
    median_age=("age", "median"),
    avg_tenure=("tenure", "mean"),
    median_tenure=("tenure", "median"),
    avg_monthlycharges=("monthlycharges", "mean"),
    avg_totalcharges=("totalcharges", "mean"),
    user_count=("customerid", "count")
)

    print(churn_summary)

def feature_engineering(df):
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-aged", "Senior"])
    df["tenure_group"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1 year", "1-2 years", "2-4 years", "4+ years"])
    return df
def plot_data(df):
    plt.figure(figsize=(10, 6))
    df["churn"].value_counts().plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.savefig(os.path.join(FIGURES, "churn_distribution.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["age"], df["totalcharges"])
    plt.xlabel("Age")
    plt.ylabel("Total Charges")
    plt.title("Age vs Total Charges")
    plt.savefig(os.path.join(FIGURES, "age_vs_totalcharges.png"))
    plt.show()

def output_data(df):
    df.to_csv(os.path.join(OUTPUT, "cleaned_data.csv"), index=False)
def main():
    mkdir()
    df = load_data()
    read_data(df)
    df = clean_data(df)
    read_data(df)
    df = feature_engineering(df)
    analysis_data(df)
    plot_data(df)
    output_data(df)
if __name__ == "__main__":
    main()