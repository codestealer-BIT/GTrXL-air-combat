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
y_3=[0.2,0.2,0.2,0.3,0.2,0.2,0.3,0.4,0.4,0.6,0.9,0.6,0.5,0.4,0.7,0.7,0.6,0.7,0.5,0.5,0.8,0.5,0.4,0.4,0.6,0.3,0,0.1,0.3,0.5,0,0.1,0,0,0.4,0.4,0.5,0.8,0.4,0,0.4,0.3,0.3,0.4,0.2,0.1,0.4,0.1,0.6,0.4]
z_3=[-1179.95, -1335.77, -1247.3, -1410.43, -1184.76, -1269.26, -1292.79, -1424.22, -858.34, -435.73, -959.39, -999.38, -991.45, -819.36, -541.35, -653.09, -686.63, -2121.91, -1069.21, -1086.37, -1105.27, -1030.41, -1496.66, -1457.63, -1292.21, -1165.61, -1193.57, -1201.85, -1329.45, -1133.88, -1167.93, -1200.53, -1232.42, -1354.53, -1316.59, -1328.11, -2648.6, -1759.72, -1335.47, -1429.7, -1145.98, -1373.66, -1407.2, -1565.48, -1565.76, -1127.88, -1498.9, -2174.05, -1752.19, -1324.67]

# 定义长度为 10 的平均滤波器
window = np.ones(10) / 10
# 使用卷积计算滑动平均
y_1_smooth = np.convolve(y_1, window, mode='valid')
y_2_smooth = np.convolve(y_2, window, mode='valid')
y_3_smooth = np.convolve(y_3, window, mode='valid')
x = np.array(list(range(0, len(y_1_smooth))))
# 示例数据
# 创建折线图
plt.plot(x, y_1_smooth, marker='o', linestyle='-', color='b', label='eps=0',markersize=2)
plt.plot(x, y_2_smooth, marker='o', linestyle='-', color='r', label='eps=0.05',markersize=2)
plt.plot(x, y_3_smooth, marker='o', linestyle='-', color='orange', label='greediness increasing',markersize=2)
shadow_width_1 = 0.05  # 阴影宽度
shadow_width_2=0.1
y_1_upper = y_1_smooth + shadow_width_1
y_1_lower = y_1_smooth - shadow_width_1
y_2_upper = y_2_smooth + shadow_width_2
y_2_lower = y_2_smooth - shadow_width_2
y_3_upper = y_3_smooth + shadow_width_2
y_3_lower = y_3_smooth - shadow_width_2
# 填充阴影
plt.fill_between(x, y_1_upper, y_1_lower, color='b', alpha=0.1)
plt.fill_between(x, y_2_upper, y_2_lower, color='r', alpha=0.1)
plt.fill_between(x, y_3_upper, y_3_lower, color='orange', alpha=0.1)#alpha是不透明度
# 添加标题和标签
plt.title('Win Rate Curve of Different Greedy Degrees of DQN')
plt.xlabel('Iteration')
plt.ylabel('Win_rate')
# 添加网格
plt.grid(True)
# 显示图例
plt.legend(loc='lower right',prop={'size':3})
# 显示图表
plt.show()

