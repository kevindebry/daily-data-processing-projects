import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_PATH = "cs_students.csv"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

def make_dirs():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    os.makedirs(FIGURE_DIR,exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def check_data(df):
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
    print(df.duplicated().sum())
    print(df.describe())

    object_cols = df.select_dtypes(include = "object").columns

    for col in object_cols:
        print(df[col].describe())
        print(df[col].value_counts().head(10))

def clean_data(df):
    
    df = df.drop_duplicates()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ","_")
    )
    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col]=df[col].str.strip()
    
    df = df[(df["gpa"]>=0)&(df["gpa"]<=4.0)]
    df = df[(df["age"]>=15)&(df["age"]<=50)]
    print("清洗完成")
    print(df.head())
    return df

def feature_engineering(df):
    skill_map={
        "Weak":1,
        "Average":2,
        "Strong":3
    }
    df["python_score"]=df["python"].map(skill_map)
    df["sql_score"]=df["sql"].map(skill_map)
    df["java_score"]=df["java"].map(skill_map)
    df["avg_skill_score"]=df[["python_score","sql_score","java_score"]].mean(axis=1)

    def get_gpa_level(gpa):
        if gpa>=3.7:
            return "high"
        elif gpa>=3.4:
            return "medium"
        else :
            return "low"
    df["gpa_level"]=df["gpa"].apply(get_gpa_level)
    print(df[[
        "student_id",
        "name",
        "gpa",
        "gpa_level",
        "python",
        "python_score",
        "sql",
        "sql_score",
        "java",
        "java_score",
        "avg_skill_score"
    ]].head())
    return df

def analyse_data(df):
    print(df["gpa"].mean())
    print(df["gpa"].max())
    print(df["gpa"].min())
    print(df["gpa"].median())

    top10_student = df.sort_values("gpa",ascending = False).head(10)
    print(top10_student[["student_id","name","gpa"]])

    career_count = df["future_career"].value_counts()
    print(career_count)
    gender_gpa = df.groupby("gender").agg(
        student_count=("student_id","count"),
        avg_gpa=("gpa","mean")
    )
    print(gender_gpa)
    top_gpa_gender = df.sort_values(["gpa","gender"],ascending = [True,False]).groupby("gender").head(10)
    print(top_gpa_gender[["gender","gpa"]])
def main():

    #make_dirs()

    df = load_data()

    check_data(df)

    df = clean_data(df)

    df = feature_engineering(df)

    df = analyse_data(df)

if __name__=="__main__":
    main()