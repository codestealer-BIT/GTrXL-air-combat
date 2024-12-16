import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
"""
missile 4500 
DQN eps=0 y_1
DQN eps=0.05 y_2
"""
# 设置中文字体
import numpy as np

# 示例数据

y_1 = np.array([0.2, 0.3, 0.4, 0.5, 0.2, 0, 0.6, 0.2, 0.8])
y_1 = np.concatenate((y_1, np.full(41, 1)))
y_2 = np.array([2, 2, 0, 2, 0, 3, 2, 3, 3, 2, 0, 3, 6, 6, 1, 1, 1, 5, 2, 0, 0, 3, 3, 4, 7, 9, 9, 8, 6, 4, 6, 8, 7, 4, 2, 3, 1, 1, 0, 2, 5, 0, 1, 2, 3, 8, 4, 0, 3, 2])
y_2 = np.array([i * 0.1 for i in y_2])
# 定义长度为 10 的平均滤波器
window = np.ones(10) / 10
# 使用卷积计算滑动平均
y_1_smooth = np.convolve(y_1, window, mode='valid')
y_2_smooth = np.convolve(y_2, window, mode='valid')
x = np.array(list(range(0, len(y_1_smooth))))
# 示例数据
# 创建折线图
plt.plot(x, y_1_smooth, marker='o', linestyle='-', color='b', label='eps=0',markersize=2)
plt.plot(x, y_2_smooth, marker='o', linestyle='-', color='r', label='eps=0.05',markersize=2)
shadow_width_1 = 0.05  # 阴影宽度
shadow_width_2=0.05
y_1_upper = y_1_smooth + shadow_width
y_1_lower = y_1_smooth - shadow_width
y_2_upper = y_2_smooth + shadow_width
y_2_lower = y_2_smooth - shadow_width

# 填充阴影
plt.fill_between(x, y_1_upper, y_1_lower, color='b', alpha=0.05)
plt.fill_between(x, y_2_upper, y_2_lower, color='r', alpha=0.05)
# 添加标题和标签
plt.title('Win Rate Curve of Different Greedy Degrees of DQN')
plt.xlabel('Iteration')
plt.ylabel('Win_rate')
# 添加网格
plt.grid(True)
# 显示图例
plt.legend(loc='lower right')
# 显示图表
plt.show()

