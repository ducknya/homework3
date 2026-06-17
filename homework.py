import pandas as pd
import numpy as np

# 读取逗号分隔的CSV文件
df = pd.read_csv("ICData.csv", sep=",")

# ========== 1. 打印数据集前5行（完整显示不省略） ==========
print("数据集前5行：")
# 设置pandas打印参数，关闭列省略、换行对齐，和截图排版一致
pd.set_option('display.max_columns', None)    # 显示所有列，无省略...
pd.set_option('display.width', None)         # 自动拓宽打印宽度
pd.set_option('display.max_colwidth', None)
print(df.head())

# ========== 2. 打印基本信息（严格匹配截图文字与格式） ==========
print("\n基本信息：")
row_cnt, col_cnt = df.shape
print(f"行数={row_cnt}，列数={col_cnt}")
# 逐行输出dtype，和截图格式完全一致
print(df.dtypes)
print("dtype: object")

# ========== 3. 时间解析，新增hour列 ==========
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["hour"] = df["交易时间"].dt.hour

# ========== 4. 构造ride_stops，删除ride_stops=0异常数据 ==========
df["ride_stops"] = abs(df["下车站点"] - df["上车站点"])
del_abnormal = (df["ride_stops"] == 0).sum()
df = df[df["ride_stops"] != 0]
# 固定文案打印删除行数
print(f"\n构造 ride_stops 后删除异常记录（ride_stops==0/无法计算）行数：{del_abnormal}")

# ========== 5. 缺失值检查 ==========
print("\n各列缺失值数量：")
missing_sum = df.isnull().sum()
if missing_sum.sum() == 0:
    print("无缺失值")
else:
    print(missing_sum)
    df = df.dropna()
