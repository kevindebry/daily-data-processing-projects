import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "supply_chain_data.csv"
OUTPUT = "output"
FIGURES = "figures"


def get_text_columns(df):
    return [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
    ]


def make_dirs():
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("数据读取完成")
    return df


def read_data(df, name="数据概览"):
    print(f"\n========== {name} ==========")
    print("前 5 行：")
    print(df.head())
    print("\n数据规模：")
    print(df.shape)
    print("\n字段类型：")
    print(df.dtypes)
    print("\n重复行数量：")
    print(df.duplicated().sum())
    print("\n缺失值统计：")
    print(df.isna().sum())
    print("\n数值字段描述：")
    print(df.describe())

    object_cols = get_text_columns(df)
    for col in object_cols:
        print(f"\n------ {col} ------")
        print(df[col].describe())
        print(df[col].value_counts().head(10))


def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    text_cols = get_text_columns(df)
    for col in text_cols:
        df[col] = df[col].str.strip()

    numeric_cols = [
        "price",
        "availability",
        "number_of_products_sold",
        "revenue_generated",
        "stock_levels",
        "lead_times",
        "order_quantities",
        "shipping_times",
        "shipping_costs",
        "lead_time",
        "production_volumes",
        "manufacturing_lead_time",
        "manufacturing_costs",
        "defect_rates",
        "costs",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    strict_positive_cols = [
        "price",
        "number_of_products_sold",
        "order_quantities",
    ]
    for col in strict_positive_cols:
        if col in df.columns:
            df = df[df[col] > 0]

    non_negative_cols = [
        "availability",
        "revenue_generated",
        "stock_levels",
        "lead_times",
        "shipping_times",
        "shipping_costs",
        "lead_time",
        "production_volumes",
        "manufacturing_lead_time",
        "manufacturing_costs",
        "defect_rates",
        "costs",
    ]
    for col in non_negative_cols:
        if col in df.columns:
            df = df[df[col] >= 0]

    if "availability" in df.columns:
        df = df[df["availability"].between(0, 100)]

    print("\n数据清洗完成")
    print(df.head())
    return df


def feature_engineering(df):
    df = df.copy()

    df["sales_value"] = df["price"] * df["number_of_products_sold"]
    df["order_value"] = df["price"] * df["order_quantities"]
    df["stock_gap"] = df["stock_levels"] - df["order_quantities"]
    df["stock_turnover_rate"] = df["number_of_products_sold"] / (df["stock_levels"] + 1)
    df["shipping_cost_per_unit"] = df["shipping_costs"] / df["order_quantities"]
    df["revenue_per_unit_sold"] = df["revenue_generated"] / df["number_of_products_sold"]
    df["total_lead_time"] = (
        df["lead_times"] + df["shipping_times"] + df["manufacturing_lead_time"]
    )
    df["total_cost_proxy"] = (
        df["shipping_costs"] + df["manufacturing_costs"] + df["costs"]
    )
    df["profit_proxy"] = df["revenue_generated"] - df["total_cost_proxy"]

    df["stock_status"] = pd.cut(
        df["stock_gap"],
        bins=[-float("inf"), -1, 20, float("inf")],
        labels=["shortage", "balanced", "surplus"],
    )
    df["defect_level"] = pd.cut(
        df["defect_rates"],
        bins=[-0.01, 1, 3, float("inf")],
        labels=["low", "medium", "high"],
    )

    print("\n特征工程完成")
    print(
        df[
            [
                "sku",
                "product_type",
                "sales_value",
                "order_value",
                "stock_status",
                "defect_level",
                "profit_proxy",
            ]
        ].head()
    )
    return df


def analysis_data(df):
    print("\n========== 数据分析 ==========")

    top10_revenue = df.sort_values("revenue_generated", ascending=False).head(10)
    print("\n收入最高的 10 个 SKU：")
    print(top10_revenue[["sku", "product_type", "revenue_generated", "profit_proxy"]])

    product_summary = df.groupby("product_type").agg(
        sku_count=("sku", "count"),
        total_sold=("number_of_products_sold", "sum"),
        avg_price=("price", "mean"),
        total_revenue=("revenue_generated", "sum"),
        avg_profit_proxy=("profit_proxy", "mean"),
        avg_defect_rate=("defect_rates", "mean"),
        avg_lead_time=("total_lead_time", "mean"),
    )
    print("\n按产品类型汇总：")
    print(product_summary)

    supplier_summary = df.groupby("supplier_name").agg(
        sku_count=("sku", "count"),
        total_revenue=("revenue_generated", "sum"),
        avg_manufacturing_cost=("manufacturing_costs", "mean"),
        avg_defect_rate=("defect_rates", "mean"),
        avg_production_volume=("production_volumes", "mean"),
    )
    print("\n按供应商汇总：")
    print(supplier_summary)

    shipping_summary = df.groupby(["shipping_carriers", "transportation_modes"]).agg(
        order_count=("sku", "count"),
        avg_shipping_time=("shipping_times", "mean"),
        avg_shipping_cost=("shipping_costs", "mean"),
        avg_route_cost=("costs", "mean"),
    )
    print("\n按承运商和运输方式汇总：")
    print(shipping_summary)

    route_summary = df.groupby("routes").agg(
        route_count=("sku", "count"),
        avg_cost=("costs", "mean"),
        avg_shipping_time=("shipping_times", "mean"),
        avg_defect_rate=("defect_rates", "mean"),
    )
    print("\n按路线汇总：")
    print(route_summary)

    product_summary.to_csv(os.path.join(OUTPUT, "product_type_summary.csv"), encoding="utf-8-sig")
    supplier_summary.to_csv(os.path.join(OUTPUT, "supplier_summary.csv"), encoding="utf-8-sig")
    shipping_summary.to_csv(os.path.join(OUTPUT, "shipping_summary.csv"), encoding="utf-8-sig")
    route_summary.to_csv(os.path.join(OUTPUT, "route_summary.csv"), encoding="utf-8-sig")


def plot_data(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "price_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    product_revenue = df.groupby("product_type")["revenue_generated"].sum().sort_values()
    product_revenue.plot(kind="barh", color="#4C78A8")
    plt.title("Revenue by Product Type")
    plt.xlabel("Revenue Generated")
    plt.ylabel("Product Type")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "revenue_by_product_type.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    supplier_defect = df.groupby("supplier_name")["defect_rates"].mean().sort_values()
    supplier_defect.plot(kind="bar", color="#F58518")
    plt.title("Average Defect Rate by Supplier")
    plt.xlabel("Supplier")
    plt.ylabel("Average Defect Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "defect_rate_by_supplier.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="manufacturing_costs",
        y="profit_proxy",
        hue="product_type",
    )
    plt.title("Manufacturing Cost vs Profit Proxy")
    plt.xlabel("Manufacturing Costs")
    plt.ylabel("Profit Proxy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "manufacturing_cost_vs_profit.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="transportation_modes",
        y="costs",
        hue="transportation_modes",
        palette="Set2",
        legend=False,
    )
    plt.title("Transportation Cost by Mode")
    plt.xlabel("Transportation Mode")
    plt.ylabel("Route Cost")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "transportation_cost_by_mode.png"), dpi=150)
    plt.close()

    print("\n图表已保存到 figures 文件夹")


def output_data(df):
    output_path = os.path.join(OUTPUT, "cleaned_supply_chain_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n清洗后的数据已保存到：{output_path}")


def main():
    make_dirs()
    df = load_data()
    read_data(df, "原始数据概览")
    df = clean_data(df)
    read_data(df, "清洗后数据概览")
    df = feature_engineering(df)
    analysis_data(df)
    plot_data(df)
    output_data(df)


if __name__ == "__main__":
    main()
