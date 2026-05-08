import pandas as pd
import matplotlib.pyplot as plt
import os

PATH_DATA = "healthcare_dataset.csv"
OUTPUT = "output"
FIGURES = "figures"
REQUIRED_COLUMNS = [
    "Name",
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Date of Admission",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Billing Amount",
    "Room Number",
    "Admission Type",
    "Discharge Date",
    "Medication",
    "Test Results",
]

def makedirs():
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

def load_data():
    if not os.path.exists(PATH_DATA):
        raise FileNotFoundError(f"Data file not found: {PATH_DATA}")

    df = pd.read_csv(PATH_DATA)
    validate_columns(df)
    return df

def validate_columns(df):
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
        )

def read_data(df):
    print("First 5 rows:")
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nShape:")
    print(df.shape)
    print("\nDuplicate rows:")
    print(df.duplicated().sum())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(df.describe())
    cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cols:
        print(f"\n{col} summary:")
        print(df[col].describe())

def clean_data(df):
    df = df.copy()

    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype("string").str.strip()

    name_like_cols = ["Name", "Doctor", "Hospital"]
    for col in name_like_cols:
        df[col] = df[col].str.title()

    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Room Number"] = pd.to_numeric(df["Room Number"], errors="coerce")

    before_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_rows - len(df)

    df = df.dropna(
        subset=[
            "Date of Admission",
            "Discharge Date",
            "Age",
            "Room Number",
            "Medical Condition",
        ]
    )
    df = df[(df["Age"] >= 0) & (df["Discharge Date"] >= df["Date of Admission"])]

    negative_billing = df["Billing Amount"] < 0
    negative_billing_count = int(negative_billing.sum())
    df.loc[negative_billing, "Billing Amount"] = pd.NA
    condition_median_billing = df.groupby("Medical Condition")[
        "Billing Amount"
    ].transform("median")
    df["Billing Amount"] = df["Billing Amount"].fillna(condition_median_billing)
    df["Billing Amount"] = df["Billing Amount"].fillna(df["Billing Amount"].median())
    df = df.dropna(subset=["Billing Amount"])

    df["Age"] = df["Age"].astype(int)
    df["Room Number"] = df["Room Number"].astype(int)

    df.to_csv(os.path.join(OUTPUT, "cleaned_healthcare_dataset.csv"), index=False)
    print(
        f"\nData cleaned: removed {duplicates_removed} duplicate rows "
        f"and fixed {negative_billing_count} negative billing values."
    )
    return df

def feature_engineering(df):
    df = df.copy()

    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    df["Admission Year"] = df["Date of Admission"].dt.year
    df["Admission Month"] = df["Date of Admission"].dt.month
    df["Admission Year Month"] = df["Date of Admission"].dt.to_period("M").astype(str)
    df["Cost Per Day"] = df["Billing Amount"] / df["Length of Stay"].clip(lower=1)

    age_bins = [-1, 17, 35, 50, 65, float("inf")]
    age_labels = ["0-17", "18-35", "36-50", "51-65", "66+"]
    df["Age Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)

    billing_labels = ["Low", "Medium", "High"]
    df["Billing Level"] = pd.qcut(
        df["Billing Amount"].rank(method="first"),
        q=3,
        labels=billing_labels
    )

    df.to_csv(os.path.join(OUTPUT, "featured_healthcare_dataset.csv"), index=False)
    print("Feature engineering done: added stay, date, age, and billing features.")
    return df

def analyse_data(df):
    overall = pd.DataFrame({
        "Metric": [
            "Total Records",
            "Average Age",
            "Average Billing Amount",
            "Median Billing Amount",
            "Average Length of Stay",
            "Average Cost Per Day"
        ],
        "Value": [
            len(df),
            df["Age"].mean(),
            df["Billing Amount"].mean(),
            df["Billing Amount"].median(),
            df["Length of Stay"].mean(),
            df["Cost Per Day"].mean()
        ]
    })

    condition_summary = (
        df.groupby("Medical Condition")
        .agg(
            Patient_Count=("Name", "count"),
            Average_Age=("Age", "mean"),
            Average_Billing=("Billing Amount", "mean"),
            Average_Length_Of_Stay=("Length of Stay", "mean"),
            Abnormal_Test_Rate=("Test Results", lambda s: (s == "Abnormal").mean())
        )
        .sort_values("Patient_Count", ascending=False)
        .reset_index()
    )

    admission_summary = (
        df.groupby("Admission Type")
        .agg(
            Patient_Count=("Name", "count"),
            Average_Billing=("Billing Amount", "mean"),
            Average_Length_Of_Stay=("Length of Stay", "mean")
        )
        .sort_values("Patient_Count", ascending=False)
        .reset_index()
    )

    insurance_summary = (
        df.groupby("Insurance Provider")
        .agg(
            Patient_Count=("Name", "count"),
            Total_Billing=("Billing Amount", "sum"),
            Average_Billing=("Billing Amount", "mean")
        )
        .sort_values("Total_Billing", ascending=False)
        .reset_index()
    )

    monthly_admissions = (
        df.groupby("Admission Year Month")
        .size()
        .rename("Admissions")
        .reset_index()
    )

    test_result_summary = (
        df.groupby(["Medical Condition", "Test Results"])
        .size()
        .rename("Patient_Count")
        .reset_index()
    )

    outputs = {
        "overall_summary.csv": overall,
        "condition_summary.csv": condition_summary,
        "admission_summary.csv": admission_summary,
        "insurance_summary.csv": insurance_summary,
        "monthly_admissions.csv": monthly_admissions,
        "test_result_summary.csv": test_result_summary
    }

    for filename, result in outputs.items():
        result.to_csv(os.path.join(OUTPUT, filename), index=False)

    print("Analysis done: summary files saved to output.")
    return outputs

def visualise_data(df):
    plt.style.use("seaborn-v0_8-whitegrid")

    condition_counts = df["Medical Condition"].value_counts()
    plt.figure(figsize=(8, 8))
    condition_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    plt.ylabel("")
    plt.title("Patients by Medical Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "patients_by_condition.png"), dpi=150)
    plt.close()

    billing_by_condition = (
        df.groupby("Medical Condition")["Billing Amount"]
        .mean()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 5))
    billing_by_condition.plot(kind="bar", color="#f58518")
    plt.title("Average Billing Amount by Medical Condition")
    plt.xlabel("Medical Condition")
    plt.ylabel("Average Billing Amount")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "average_billing_by_condition.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    df["Age"].plot(kind="hist", bins=20, color="#54a24b", edgecolor="white")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "age_distribution.png"), dpi=150)
    plt.close()

    admission_counts = df["Admission Type"].value_counts()
    plt.figure(figsize=(7, 7))
    admission_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    plt.ylabel("")
    plt.title("Patients by Admission Type")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "patients_by_admission_type.png"), dpi=150)
    plt.close()

    monthly_admissions = df.groupby("Admission Year Month").size()
    plt.figure(figsize=(12, 5))
    monthly_admissions.plot(kind="line", marker="o", color="#e45756")
    plt.title("Monthly Admissions")
    plt.xlabel("Admission Year Month")
    plt.ylabel("Admissions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "monthly_admissions.png"), dpi=150)
    plt.close()

    test_results = pd.crosstab(df["Medical Condition"], df["Test Results"])
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 8))
    axes = axes.flatten()
    for ax, condition in zip(axes, test_results.index):
        counts = test_results.loc[condition]
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
        )
        ax.set_title(condition)
        ax.axis("equal")

    for ax in axes[len(test_results.index):]:
        ax.axis("off")

    fig.suptitle("Test Results by Medical Condition")
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES, "test_results_by_condition.png"), dpi=150)
    plt.close()

    print("Visualization done: charts saved to figures.")

def main():
    makedirs()
    df = load_data()
    read_data(df)
    df = clean_data(df)
    df = feature_engineering(df)
    analyse_data(df)
    visualise_data(df)

if __name__ == "__main__":
    main()
