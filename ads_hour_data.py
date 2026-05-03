import os

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "ads_hour.csv"
FIGURE_DIR = "figures"

def main():
    os.makedirs(FIGURE_DIR,exist_ok=True)

    #1读取数据
    df = pd.read_csv(DATA_PATH)
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
    print(df.duplicated().sum())

    #2时间段字符处理
    df["Date"] = pd.to_datetime(df["Date"],format= "%m/%d/%y %H:%M")

    #3提取时间特征
    df["date"] = df["Date"].dt.date
    df["hour"]= df["Date"].dt.hour
    df["weekday"]=df["Date"].dt.day_name()
    df["month"] = df["Date"].dt.month

    print(df.head())
    print(df["ads"].describe())

    #4按照小时平均ads
    hourly_avg = df.groupby("hour")["ads"].mean()
    print(hourly_avg)

    #5按照日期总量
    daily_total = df.groupby("date")["ads"].sum()
    print(daily_total.head())

    #6按照星期
    weekday_avg = df.groupby("weekday")["ads"].mean()
    print(weekday_avg)

    #7找出ads最高的前十个小时
    top10_hours = df.sort_values("ads",ascending=False).head(10)
    print(top10_hours[["Date","ads","hour","weekday"]])


     #8画图1：每天总ads趋势
    plt.figure(figsize = (12,5))
    daily_total.plot()
    plt.title("Daily Total Ads")
    plt.xlabel("Date")
    plt.ylabel("Total Ads")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "daily_total_ads.png"))
    plt.show()

    # 9. 画图2：每小时平均 ads
    plt.figure(figsize=(10, 5))
    hourly_avg.plot(kind="bar")
    plt.title("Average Ads by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Ads")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "hourly_average_ads.png"))
    plt.show()

    #10保存清洗后数据
    df.to_csv("cleaned_ads_hour.csv", index=False, encoding="utf-8-sig")

    print("\n分析完成，图片已保存到 figures/ 文件夹。")
if __name__=="__main__":

   
    main()