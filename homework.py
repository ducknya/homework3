import pandas as pd
import numpy as np

# ========== 1. 读取数据 ==========
# 读取csv数据集
df = pd.read_csv("ICData.csv")

# 打印数据集前5行
print("数据集前5行：")
print(df.head())

# 打印数据集基础信息（行数、列数、各字段数据类型）
print("\n数据集基本信息：")
print(df.info())


# ========== 2. 时间解析 ==========
# 将字符串格式的交易时间转换为pandas datetime类型，自动适配格式
df["交易时间"] = pd.to_datetime(df["交易时间"])

# 从时间中提取小时（整数），新增hour列
df["hour"] = df["交易时间"].dt.hour


# ========== 3. 构造衍生字段 ride_stops ==========
# 计算搭乘站点数：下车站点与上车站点的绝对差值
df["ride_stops"] = (df["下车站点"] - df["上车站点"]).abs()

# 统计ride_stops为0的异常记录行数
abnormal_count = len(df[df["ride_stops"] == 0])
# 删除异常记录
df = df[df["ride_stops"] != 0]

print(f"\n删除ride_stops为0的异常记录行数：{abnormal_count}")


# ========== 4. 缺失值检查与处理 ==========
# 统计并打印每列的缺失值数量
print("\n各列缺失值数量：")
missing_num = df.isnull().sum()
print(missing_num)

# 删除包含缺失值的所有记录
df = df.dropna()

# 输出清洗后数据集的规模，验证结果
print(f"\n缺失值处理后数据集规模：共 {df.shape[0]} 行，{df.shape[1]} 列")
