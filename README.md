# 史浩楠-25361138-第三次人工智能编程作业
仓库链接: (https://github.com/ducknya/homework3)
## 1. 任务拆解与 AI 协作策略
将任务分开，先让AI读取数据，同时利用图片辅助确定输出的结果
## 2. 核心 Prompt 迭代记录
 初代 Prompt：print(f"最大15分钟刷卡量...")import pandas as pd
 AI 生成的问题：运行直接报 SyntaxError: invalid syntax，报错定位在 print 行末尾。Python 按行解析代码，一行内默认只能有一条独立语句，直接拼接两条语句会触发语法错误。
## 3. Debug 记录
 报错现象：Traceback (most recent call last):
  File "pandas/_libs/tslibs/offsets.pyx", line 6213, in pandas._libs.tslibs.offsets._get_offset
KeyError: 'T'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "pandas/_libs/tslibs/offsets.pyx", line 6344, in pandas._libs.tslibs.offsets.to_offset
  File "pandas/_libs/tslibs/offsets.pyx", line 6219, in pandas._libs.tslibs.offsets._get_offset
  File "pandas/_libs/tslibs/offsets.pyx", line 6137, in pandas._libs.tslibs.offsets.raise_invalid_freq
ValueError: Invalid frequency: T. Failed to parse with error message: KeyError('T'). Did you mean min?

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\ducknya\PycharmProjects\PythonProject\homework.py", line 208, in <module>
    five_min_flow = peak_hour_records.resample('5T').size()
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Users\ducknya\PycharmProjects\PythonProject\.venv\Lib\site-packages\pandas\core\generic.py", line 9423, in resample
    return get_resampler(
        cast("Series | DataFrame", self),
    ...<8 lines>...
        group_keys=group_keys,
    )
  File "C:\Users\ducknya\PycharmProjects\PythonProject\.venv\Lib\site-packages\pandas\core\resample.py", line 2334, in get_resampler
    tg = TimeGrouper(obj, **kwds)  # type: ignore[arg-type]
  File "C:\Users\ducknya\PycharmProjects\PythonProject\.venv\Lib\site-packages\pandas\core\resample.py", line 2420, in __init__
    freq = to_offset(freq)
  File "pandas/_libs/tslibs/offsets.pyx", line 6229, in pandas._libs.tslibs.offsets.to_offset
  File "pandas/_libs/tslibs/offsets.pyx", line 6352, in pandas._libs.tslibs.offsets.to_offset
  File "pandas/_libs/tslibs/offsets.pyx", line 6137, in pandas._libs.tslibs.offsets.raise_invalid_freq
ValueError: Invalid frequency: 5T. Failed to parse with error message: ValueError("Invalid frequency: T. Failed to parse with error message: KeyError('T'). Did you mean min?") 
解决过程：错误是高版本 pandas（2.2+）的语法变更导致的：官方已经弃用了 T 作为分钟频率的缩写，新版里分钟的标准写法是 min。
只需要把代码里两处 resample 的频率参数改一下即可：
'5T' → '5min'
'15T' → '15min'
## 4. 人工代码审查（逐行中文注释） 
（# ===================== 任务4 高峰小时系数PHF计算 =====================
print("\n[任务4] 高峰小时系数PHF计算结果：")

# ========== 步骤1：识别高峰小时（以上车刷卡量为统计口径） ==========
hourly_flow = df_board.groupby(df_board["交易时间"].dt.hour).size()
peak_hour = hourly_flow.idxmax()
peak_hour_total = hourly_flow[peak_hour]

peak_start = f"{peak_hour:02d}:00"
peak_end = f"{peak_hour + 1:02d}:00"
print(f"高峰小时：{peak_start} ~ {peak_end}，刷卡量：{peak_hour_total} 次")

# ========== 步骤2：筛选高峰小时内的全部上车记录 ==========
peak_hour_records = df_board[df_board["交易时间"].dt.hour == peak_hour].copy()
peak_hour_records = peak_hour_records.set_index("交易时间")

# ========== 步骤3：5分钟粒度统计 + PHF5计算 ==========
# 修复：新版pandas分钟频率用 min 代替 T
five_min_flow = peak_hour_records.resample('5min').size()
max_5min_value = five_min_flow.max()
max_5min_start_time = five_min_flow.idxmax()

max_5min_start_str = max_5min_start_time.strftime('%H:%M')
max_5min_end_str = (max_5min_start_time + pd.Timedelta(minutes=5)).strftime('%H:%M')

PHF5 = peak_hour_total / (12 * max_5min_value)
PHF5_str = f"{PHF5:.4f}"
print(f"最大5分钟刷卡量（{max_5min_start_str}~{max_5min_end_str}）：{max_5min_value} 次 PHF5 = {peak_hour_total} / (12 × {max_5min_value}) = {PHF5_str}")

# ========== 步骤4：15分钟粒度统计 + PHF15计算 ==========
fifteen_min_flow = peak_hour_records.resample('15min').size()
max_15min_value = fifteen_min_flow.max()
max_15min_start_time = fifteen_min_flow.idxmax()

max_15min_start_str = max_15min_start_time.strftime('%H:%M')
max_15min_end_str = (max_15min_start_time + pd.Timedelta(minutes=15)).strftime('%H:%M')

PHF15 = peak_hour_total / (4 * max_15min_value)
PHF15_str = f"{PHF15:.4f}"
print(f"最大15分钟刷卡量（{max_15min_start_str}~{max_15min_end_str}）：{max_15min_value} 次 PHF15 = {peak_hour_total} / (4 × {max_15min_value}) = {PHF15_str}")
