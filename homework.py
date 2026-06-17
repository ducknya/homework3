import pandas as pd
import numpy as np

# ========== 1. 读取数据 ==========
df = pd.read_csv("ICData.csv")

# 打印前5行，匹配案例输出格式
print("数据集前5行：")
print(df.head())


# ========== 输出原始数据基本信息 ==========
# 注意：基本信息在数据处理前输出，保证交易时间为object类型，和案例一致
print("\n基本信息：")
print(f"行数={df.shape[0]}，列数={df.shape[1]}")
print(df.dtypes)


# ========== 2. 时间解析 ==========
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["hour"] = df["交易时间"].dt.hour


# ========== 3. 构造衍生字段并删除异常记录 ==========
# 计算搭乘站点数（绝对差值）
df["ride_stops"] = (df["下车站点"] - df["上车站点"]).abs()

# 统计ride_stops为0的异常行数
del_count = (df["ride_stops"] == 0).sum()
# 删除异常记录
df = df[df["ride_stops"] != 0]

# 按案例格式打印删除行数
print(f"\n构造 ride_stops 后删除异常记录（ride_stops==0/无法计算）行数：{del_count}")


# ========== 4. 缺失值检查与处理 ==========
print("\n各列缺失值数量：")
missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("无缺失值")
else:
    print(missing_values)
    # 存在缺失值则删除对应记录
    df = df.dropna()
