import pandas as pd
import matplotlib.pyplot as plt

import os
from io import StringIO

DATA_PATH = "weatherHistory.csv"
OUTPUT = "output"
FIGURE = "figure"
DATE_COL = "formatted_date"
SUMMARY_COL = "summary"
PRECIP_COL = "precip_type"
TEMP_COL = "temperature_(c)"
APPARENT_TEMP_COL = "apparent_temperature_(c)"
HUMIDITY_COL = "humidity"
WIND_SPEED_COL = "wind_speed_(km/h)"
WIND_BEARING_COL = "wind_bearing_(degrees)"
VISIBILITY_COL = "visibility_(km)"
LOUD_COVER_COL = "loud_cover"
PRESSURE_COL = "pressure_(millibars)"
DAILY_SUMMARY_COL = "daily_summary"

METRIC_COLUMNS = [
    TEMP_COL,
    APPARENT_TEMP_COL,
    HUMIDITY_COL,
    WIND_SPEED_COL,
    WIND_BEARING_COL,
    VISIBILITY_COL,
    LOUD_COVER_COL,
    PRESSURE_COL,
]

METRIC_CN_NAMES = {
    TEMP_COL: "温度",
    APPARENT_TEMP_COL: "体感温度",
    HUMIDITY_COL: "湿度",
    WIND_SPEED_COL: "风速",
    WIND_BEARING_COL: "风向角度",
    VISIBILITY_COL: "能见度",
    LOUD_COVER_COL: "云量覆盖",
    PRESSURE_COL: "气压",
    "go_out_score": "出门适宜分",
}
#总体原则，严格按照注释进行实现
def makedirs():
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(FIGURE, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def read_data(df):
    #补充并调整函数，本模块用于数据处理者对数据有基本认知
    makedirs()
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    overview = [
        "数据基础认知报告",
        "=" * 40,
        "前5行：",
        df.head().to_string(),
        "",
        f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列",
        "",
        "字段类型：",
        df.dtypes.to_string(),
        "",
        f"重复行数量：{df.duplicated().sum()}",
        "",
        "缺失值统计：",
        df.isnull().sum().to_string(),
        "",
        "字段信息：",
        info_buffer.getvalue(),
        "",
        "数值字段描述统计：",
        df.describe().to_string(),
        "",
        "字符字段描述统计：",
    ]
    for col in object_cols:
        overview.extend(["", f"[{col}]", df[col].describe().to_string()])

    overview_text = "\n".join(overview)
    print(overview_text)
    with open(os.path.join(OUTPUT, "data_overview.txt"), "w", encoding="utf-8") as f:
        f.write(overview_text)

def clean_data(df):
    #清洗异常值，缺失值，重复值等
    #对不符合逻辑的数值和object进行处理
    #统一小写，把空格用_代替
    #把时间转化成时间格式
    #导出清洗结果
    df = df.copy()
    before_rows = len(df)
    before_duplicates = df.duplicated().sum()

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
    invalid_dates = df[DATE_COL].isnull().sum()
    df = df.dropna(subset=[DATE_COL])

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = (
            df[col]
            .fillna("unknown")
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
        df[col] = df[col].replace({"": "unknown", "nan": "unknown", "<na>": "unknown"})

    if PRECIP_COL in df.columns:
        df.loc[~df[PRECIP_COL].isin(["rain", "snow", "unknown"]), PRECIP_COL] = "unknown"

    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_numeric_counts = {}
    rules = {
        TEMP_COL: df[TEMP_COL].between(-90, 60),
        APPARENT_TEMP_COL: df[APPARENT_TEMP_COL].between(-100, 70),
        HUMIDITY_COL: df[HUMIDITY_COL].between(0, 1),
        WIND_SPEED_COL: df[WIND_SPEED_COL].ge(0),
        WIND_BEARING_COL: df[WIND_BEARING_COL].between(0, 360),
        VISIBILITY_COL: df[VISIBILITY_COL].ge(0),
        LOUD_COVER_COL: df[LOUD_COVER_COL].ge(0),
        PRESSURE_COL: df[PRESSURE_COL].between(800, 1100),
    }
    for col, valid_mask in rules.items():
        invalid_count = (~valid_mask & df[col].notnull()).sum()
        invalid_numeric_counts[col] = int(invalid_count)
        df.loc[~valid_mask, col] = pd.NA

    missing_before_fill = df.isnull().sum()
    for col in METRIC_COLUMNS:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].fillna("unknown")

    df = df.drop_duplicates()
    after_rows = len(df)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    clean_report = pd.DataFrame(
        {
            "item": [
                "raw_rows",
                "raw_duplicates",
                "invalid_dates_removed",
                "rows_after_cleaning",
                "rows_removed_total",
            ],
            "value": [
                before_rows,
                before_duplicates,
                int(invalid_dates),
                after_rows,
                before_rows - after_rows,
            ],
        }
    )
    invalid_report = pd.DataFrame(
        {
            "column": list(invalid_numeric_counts.keys()),
            "invalid_value_count": list(invalid_numeric_counts.values()),
            "missing_before_fill": [int(missing_before_fill[col]) for col in invalid_numeric_counts],
        }
    )

    df.to_csv(os.path.join(OUTPUT, "weather_cleaned.csv"), index=False, encoding="utf-8-sig")
    clean_report.to_csv(os.path.join(OUTPUT, "clean_report.csv"), index=False, encoding="utf-8-sig")
    invalid_report.to_csv(os.path.join(OUTPUT, "invalid_value_report.csv"), index=False, encoding="utf-8-sig")
    return df

def feature_engineering(df):
    #特征工程
    #分别对一下数据划分特征层级（划分原则参考气象标准）
    # Temperature (C)             float64
    # Apparent Temperature (C)    float64
    # Humidity                    float64
    # Wind Speed (km/h)           float64
    # Wind Bearing (degrees)      float64
    # Visibility (km)             float64
    # Loud Cover                  float64
    # Pressure (millibars)        float64
    #综合几点指标，构建是否适合出门的指标
    df = df.copy()

    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["date"] = df[DATE_COL].dt.date
    df["hour"] = df[DATE_COL].dt.hour

    df["temperature_level"] = pd.cut(
        df[TEMP_COL],
        bins=[-float("inf"), -10, 0, 10, 25, 35, float("inf")],
        labels=["severe_cold", "cold", "cool", "comfortable", "hot", "extreme_hot"],
    )
    df["apparent_temperature_level"] = pd.cut(
        df[APPARENT_TEMP_COL],
        bins=[-float("inf"), -10, 0, 10, 25, 35, float("inf")],
        labels=["severe_cold", "cold", "cool", "comfortable", "hot", "extreme_hot"],
    )
    df["humidity_level"] = pd.cut(
        df[HUMIDITY_COL],
        bins=[-0.01, 0.4, 0.6, 0.8, 1.0],
        labels=["dry", "comfortable", "humid", "very_humid"],
    )
    df["wind_speed_level"] = pd.cut(
        df[WIND_SPEED_COL],
        bins=[-0.01, 1, 5, 11, 19, 29, 39, 50, float("inf")],
        labels=[
            "calm",
            "light_air",
            "light_breeze",
            "gentle_breeze",
            "moderate_breeze",
            "fresh_breeze",
            "strong_wind",
            "gale_or_above",
        ],
    )
    df["wind_direction"] = df[WIND_BEARING_COL].map(_wind_direction)
    df["visibility_level"] = pd.cut(
        df[VISIBILITY_COL],
        bins=[-0.01, 0.2, 0.5, 1, 10, 20, float("inf")],
        labels=["dense_fog", "heavy_fog", "fog", "mist", "good", "very_good"],
    )
    df["loud_cover_level"] = df[LOUD_COVER_COL].map(lambda value: "none" if value == 0 else "covered")
    df["pressure_level"] = pd.cut(
        df[PRESSURE_COL],
        bins=[-float("inf"), 980, 1000, 1020, 1040, float("inf")],
        labels=["very_low", "low", "normal", "high", "very_high"],
    )

    score = pd.Series(100.0, index=df.index)
    score -= df["temperature_level"].map(
        {
            "severe_cold": 35,
            "cold": 22,
            "cool": 8,
            "comfortable": 0,
            "hot": 15,
            "extreme_hot": 35,
        }
    ).astype(float)
    score -= df["humidity_level"].map({"dry": 8, "comfortable": 0, "humid": 8, "very_humid": 18}).astype(float)
    score -= df["wind_speed_level"].map(
        {
            "calm": 0,
            "light_air": 0,
            "light_breeze": 0,
            "gentle_breeze": 3,
            "moderate_breeze": 8,
            "fresh_breeze": 16,
            "strong_wind": 28,
            "gale_or_above": 40,
        }
    ).astype(float)
    score -= df["visibility_level"].map(
        {"dense_fog": 35, "heavy_fog": 28, "fog": 22, "mist": 8, "good": 0, "very_good": 0}
    ).astype(float)
    score -= df["pressure_level"].map(
        {"very_low": 16, "low": 8, "normal": 0, "high": 4, "very_high": 10}
    ).astype(float)
    score -= df[SUMMARY_COL].str.contains("foggy", na=False).astype(int) * 18
    score -= df[SUMMARY_COL].str.contains("breezy|windy|dangerously_windy", regex=True, na=False).astype(int) * 12
    score -= df[PRECIP_COL].eq("snow").astype(int) * 10

    df["go_out_score"] = score.clip(0, 100).round(1)
    df["go_out_level"] = pd.cut(
        df["go_out_score"],
        bins=[-0.1, 50, 70, 85, 100],
        labels=["not_suitable", "normal", "suitable", "very_suitable"],
    )
    df["is_good_to_go_out"] = df["go_out_score"].ge(70)

    df.to_csv(os.path.join(OUTPUT, "weather_featured.csv"), index=False, encoding="utf-8-sig")
    return df

def analyze_data(df):
    #分析数据
    #按照年份分析整年各项指标的平均值，极值及其对应出现日期
    #找出具有强因果关系的指标
    yearly_records = []
    for year, year_df in df.groupby("year"):
        for col in METRIC_COLUMNS:
            min_idx = year_df[col].idxmin()
            max_idx = year_df[col].idxmax()
            yearly_records.append(
                {
                    "year": int(year),
                    "metric": col,
                    "metric_cn": METRIC_CN_NAMES.get(col, col),
                    "mean": round(year_df[col].mean(), 4),
                    "min": round(year_df.loc[min_idx, col], 4),
                    "min_date": year_df.loc[min_idx, DATE_COL].date(),
                    "max": round(year_df.loc[max_idx, col], 4),
                    "max_date": year_df.loc[max_idx, DATE_COL].date(),
                }
            )
    yearly_summary = pd.DataFrame(yearly_records)

    relation_cols = METRIC_COLUMNS + ["go_out_score"]
    corr = df[relation_cols].corr(numeric_only=True)
    relation_records = []
    for i, left_col in enumerate(relation_cols):
        for right_col in relation_cols[i + 1:]:
            corr_value = corr.loc[left_col, right_col]
            if abs(corr_value) >= 0.7:
                relation_records.append(
                    {
                        "metric_1": left_col,
                        "metric_1_cn": METRIC_CN_NAMES.get(left_col, left_col),
                        "metric_2": right_col,
                        "metric_2_cn": METRIC_CN_NAMES.get(right_col, right_col),
                        "correlation": round(corr_value, 4),
                        "relationship": "positive" if corr_value > 0 else "negative",
                    }
                )
    strong_relationships = pd.DataFrame(relation_records).sort_values(
        "correlation", key=lambda series: series.abs(), ascending=False
    )

    go_out_summary = (
        df.groupby("go_out_level", observed=True)
        .agg(count=("go_out_score", "size"), avg_score=("go_out_score", "mean"))
        .reset_index()
    )
    go_out_summary["avg_score"] = go_out_summary["avg_score"].round(2)

    yearly_summary.to_csv(os.path.join(OUTPUT, "yearly_metric_summary.csv"), index=False, encoding="utf-8-sig")
    corr.to_csv(os.path.join(OUTPUT, "metric_correlation_matrix.csv"), encoding="utf-8-sig")
    strong_relationships.to_csv(os.path.join(OUTPUT, "strong_relationships.csv"), index=False, encoding="utf-8-sig")
    go_out_summary.to_csv(os.path.join(OUTPUT, "go_out_summary.csv"), index=False, encoding="utf-8-sig")
    return yearly_summary, strong_relationships

def visualize_data(df):
    #可视化数据
    #根据分析数据的结果，选择合适的图表进行可视化展示，挑选原则如下：
    #因果变化类：从数据分析中得出具有强关联性质的指标，选择合适的图表进行可视化展示
    #时间变化类：按照年份分析整年各项指标的平均值，极值及其对应出现日期，选择合适的图表进行可视化展示
    #总体分布类：对整个数据时间内的数据笼统分析，得出总体分布
    #总图表数目不超过6个，并有详细中文标注
    _set_chinese_font()
    plt.close("all")

    relation_cols = METRIC_COLUMNS + ["go_out_score"]
    corr = df[relation_cols].corr(numeric_only=True)
    labels = [METRIC_CN_NAMES.get(col, col) for col in relation_cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("数值指标相关系数热力图")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="相关系数")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE, "01_correlation_heatmap.png"), dpi=180)
    plt.close(fig)

    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample_df[TEMP_COL], sample_df[APPARENT_TEMP_COL], s=8, alpha=0.25)
    ax.set_title("强关联指标散点图：温度与体感温度")
    ax.set_xlabel("温度（℃）")
    ax.set_ylabel("体感温度（℃）")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE, "02_temperature_apparent_scatter.png"), dpi=180)
    plt.close(fig)

    yearly_mean = df.groupby("year")[[
        TEMP_COL,
        APPARENT_TEMP_COL,
        HUMIDITY_COL,
        WIND_SPEED_COL,
        VISIBILITY_COL,
        PRESSURE_COL,
    ]].mean()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    yearly_plot_cols = [
        TEMP_COL,
        APPARENT_TEMP_COL,
        HUMIDITY_COL,
        WIND_SPEED_COL,
        VISIBILITY_COL,
        PRESSURE_COL,
    ]
    for ax, col in zip(axes.ravel(), yearly_plot_cols):
        ax.plot(yearly_mean.index, yearly_mean[col], marker="o", linewidth=1.8)
        ax.set_title(f"{METRIC_CN_NAMES.get(col, col)}年度平均值")
        ax.set_xlabel("年份")
        ax.set_ylabel(METRIC_CN_NAMES.get(col, col))
        ax.grid(alpha=0.25)
    fig.suptitle("按年份统计的主要气象指标平均值", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE, "03_yearly_metric_means.png"), dpi=180)
    plt.close(fig)

    yearly_temp_extreme = df.groupby("year")[TEMP_COL].agg(["min", "max"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(yearly_temp_extreme.index, yearly_temp_extreme["max"], marker="o", label="年最高温")
    ax.plot(yearly_temp_extreme.index, yearly_temp_extreme["min"], marker="o", label="年最低温")
    ax.set_title("年度温度极值变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("温度（℃）")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE, "04_yearly_temperature_extremes.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["go_out_score"], bins=20, color="#4C78A8", edgecolor="white")
    axes[0].axvline(df["go_out_score"].mean(), color="#E45756", linestyle="--", label="平均分")
    axes[0].set_title("出门适宜分总体分布")
    axes[0].set_xlabel("出门适宜分")
    axes[0].set_ylabel("记录数量")
    axes[0].legend()
    go_out_counts = df["go_out_level"].value_counts().reindex(
        ["not_suitable", "normal", "suitable", "very_suitable"]
    )
    go_out_labels = ["不适合", "一般", "适合", "很适合"]
    axes[1].bar(go_out_labels, go_out_counts.values, color="#72B7B2")
    axes[1].set_title("出门适宜等级数量")
    axes[1].set_xlabel("适宜等级")
    axes[1].set_ylabel("记录数量")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE, "05_go_out_distribution.png"), dpi=180)
    plt.close(fig)

    summary_counts = df[SUMMARY_COL].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([label.replace("_", " ") for label in summary_counts.index], summary_counts.values, color="#F58518")
    ax.set_title("天气概况出现次数 Top 10")
    ax.set_xlabel("记录数量")
    ax.set_ylabel("天气概况")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE, "06_summary_top10.png"), dpi=180)
    plt.close(fig)


def _wind_direction(degrees):
    if pd.isna(degrees):
        return "unknown"
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return directions[int(((degrees + 22.5) % 360) // 45)]


def _set_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False



def main():
    makedirs()
    df = load_data()
    read_data(df)
    df = clean_data(df)
    df = feature_engineering(df)
    analyze_data(df)
    visualize_data(df)
  


if __name__ == "__main__":
    main()
