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

# ===================== 任务2 时间分布分析 =====================
# ---------- 任务2(a) 早晚时段刷卡量统计（numpy布尔索引实现） ----------
# 筛选刷卡类型 = 0 的上车记录
df_board = df[df["刷卡类型"] == 0]
# 将小时列转为 numpy 数组，用于numpy方法统计
hour_arr = df_board["hour"].to_numpy()

# 使用 numpy 布尔索引统计两个时段刷卡量
early_count = np.sum(hour_arr < 7)       # 早峰前：hour < 7
late_count = np.sum(hour_arr >= 22)     # 深夜：hour >= 22
total_count = len(hour_arr)             # 全天刷卡类型=0的总刷卡量

# 计算占全天总量的百分比
early_pct = early_count / total_count * 100
late_pct = late_count / total_count * 100

# 按要求格式打印结果
print("\n[任务2(a)] 早峰前/深夜上车刷卡量：")
print(f"早上7点前公共交通上车刷卡量为：{early_count} 次，占全天 {early_pct:.2f}%")
print(f"晚上10点后公共交通上车刷卡量为：{late_count} 次，占全天 {late_pct:.2f}%")


# ---------- 任务2(b) 24小时刷卡量分布可视化（matplotlib） ----------
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 统计0~23每个小时的刷卡量，补全无数据的小时为0
hours_all = np.arange(24)
hour_counts = df_board.groupby("hour").size().reindex(hours_all, fill_value=0).values

# 配置柱体颜色：早峰前(<7)与深夜(≥22)用橙色高亮，其余时段用蓝色
bar_colors = []
for h in hours_all:
    if h < 7 or h >= 22:
        bar_colors.append("#ff7f0e")  # 高亮橙色
    else:
        bar_colors.append("#1f77b4")  # 常规蓝色

# 创建画布并绘制柱状图
plt.figure(figsize=(13, 7))
plt.bar(hours_all, hour_counts, color=bar_colors)

# 坐标轴设置
plt.xticks(np.arange(0, 24, 2), fontsize=12)  # x轴0~23，步长为2
plt.xlabel("Hour (0~23)", fontsize=14)
plt.ylabel("Boarding Count", fontsize=14)

# 英文标题
plt.title("24-hour Boarding Counts Distribution", fontsize=15, pad=12)

# 添加图例
legend_elem = [Patch(facecolor="#ff7f0e", label="Early & Late Highlight")]
plt.legend(handles=legend_elem, fontsize=12)

# 添加水平网格线
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 保存图像，dpi=150
plt.savefig("hour_distribution.png", dpi=150, bbox_inches="tight")

# 显示图表（可根据需要删除，不影响保存结果）
plt.show()

print("\n[任务2(b)] 已保存图像：hour_distribution.png")
