"""
missile=4500
    win_rate PPO y_1
    win_rate DQN eps=0 y_2
"""
import matplotlib.pyplot as plt
import numpy as np
y_1=[50.0, 100.0, 100.0, 100.0, 100.0, 90.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
y_1=np.array([i*0.01 for i in y_1])
y_2=[30.0, 90.0, 100.0, 30.0, 90.0, 90.0, 70.0, 60.0, 50.0, 100.0, 100.0, 100.0, 80.0, 50.0, 0.0, 10.0, 40.0, 90.0, 100.0, 100.0, 100.0, 50.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0]
y_2=np.array([i*0.01 for i in y_2])
y_3=np.ones(y_2.shape)
def plot_fig(y_1,y_2,y_3):
    # 定义长度为 10 的平均滤波器
    window = np.ones(10) / 10
    # 使用卷积计算滑动平均
    y_1_smooth = np.convolve(y_1, window, mode='valid')
    y_2_smooth = np.convolve(y_2, window, mode='valid')
    y_3_smooth = np.convolve(y_3, window, mode='valid')
    x = np.array(list(range(0, len(y_1_smooth))))
    # 示例数据
    plt.figure(dpi=250)
    # 创建折线图
    plt.plot(x, y_1_smooth, marker='o', linestyle='-', color='b', label='PPO',markersize=2)
    plt.plot(x, y_2_smooth, marker='o', linestyle='-', color='r', label='DQN',markersize=2)
    # plt.plot(x, y_3_smooth, marker='o', linestyle='-', color='orange', label='eps decreasing',markersize=2)

    shadow_width_1 = 0.05  # 阴影宽度
    y_1_upper = y_1_smooth + shadow_width_1
    y_1_lower = y_1_smooth - shadow_width_1
    y_2_upper = y_2_smooth + shadow_width_1
    y_2_lower = y_2_smooth - shadow_width_1
    y_3_upper = y_3_smooth + shadow_width_1
    y_3_lower = y_3_smooth - shadow_width_1
    # 填充阴影
    plt.fill_between(x, y_1_upper, y_1_lower, color='b', alpha=0.1)
    plt.fill_between(x, y_2_upper, y_2_lower, color='r', alpha=0.1)
    # plt.fill_between(x, y_3_upper, y_3_lower, color='orange', alpha=0.1)#alpha是不透明度
    # 添加标题和标签
    plt.title('Win_rate Curve of Different Greedy Degrees of DQN')
    plt.xlabel('Iteration(moving average)')
    plt.ylabel('Win_rate')
    # 添加网格
    plt.grid(True)
    # 显示图例
    plt.legend(loc='lower right',prop={'size':6})
    # 显示图表
    plt.show()
if __name__=="__main__":
    plot_fig(y_1,y_2,y_3)