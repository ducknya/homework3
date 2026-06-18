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

# 修复：绘图前清理所有画布，避免多次运行图形叠加
plt.close('all')

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
# 修复：移除 plt.show()，消除阻塞，代码可连续执行

print("\n[任务2(b)] 已保存图像：hour_distribution.png")


# ===================== 任务3 线路站点分析 =====================
# ---------- 3(a) 定义统计函数并打印结果 ----------
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 按线路分组，计算搭乘站点数的均值与标准差
    route_stats = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    # 重命名列名，与要求的输出列一致
    route_stats.columns = [route_col, 'mean_stops', 'std_stops']
    # 按平均站点数降序排序并重置索引
    route_stats = route_stats.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return route_stats

# 调用函数获取全线路统计结果
route_result = analyze_route_stops(df)

# 打印前10行统计结果
print("\n[任务3] 每条线路的平均搭乘站点数及标准差（前10行）：")
print(route_result.head(10))


# ---------- 3(b) Top15线路水平条形图可视化 ----------
import seaborn as sns

# 修复：绘图前再次清理画布，确保与任务2的图完全独立
plt.close('all')

# 提取平均站点数最高的前15条线路
top15_routes = route_result.head(15)

plt.figure(figsize=(10, 8))
# 修复：绑定hue并关闭图例，消除FutureWarning
sns.barplot(
    data=top15_routes,
    y='线路号',
    x='mean_stops',
    hue='线路号',
    palette='Blues_d',
    orient='h',
    legend=False
)

# 手动添加标准差误差棒
y_positions = np.arange(len(top15_routes))
plt.errorbar(
    x=top15_routes['mean_stops'],
    y=y_positions,
    xerr=top15_routes['std_stops'],
    fmt='none',
    ecolor='black',
    capsize=0.3
)

# 图表标注设置
plt.title('Top 15 Routes: Mean Ride Stops (with Std Dev)', fontsize=13, pad=12)
plt.xlabel('Mean Ride Stops', fontsize=11)
plt.ylabel('Route ID', fontsize=11)
# x轴从0起始
plt.xlim(left=0)
# 添加垂直网格线
plt.grid(axis='y', linestyle='-', alpha=0.7)

# 保存图像，dpi=150
plt.savefig('route_stops.png', dpi=150, bbox_inches='tight')
# 修复：移除 plt.show()，消除阻塞

print("\n[任务3] 已保存图像：route_stops.png")
