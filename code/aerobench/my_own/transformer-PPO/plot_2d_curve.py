import matplotlib.pyplot as plt

def plot_2d_curve(x_values, y_values, title="2D Curve Plot"):
    """
    绘制二维曲线图，并用箭头标记起点和终点，同时显示其 (x, y) 坐标。
    
    :param x_values: X 坐标列表
    :param y_values: Y 坐标列表
    :param title: 图表标题
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制曲线
    plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=3, label="Path", color='b')

    # 统一箭头参数
    arrow_params = dict(arrowstyle="->", lw=1.5, color='black')

    # 获取起点和终点坐标
    start_x, start_y = x_values[0], y_values[0]
    end_x, end_y = x_values[-1], y_values[-1]

    # 调整标注位置，使其不会遮挡数据点
    offset_x = (max(x_values) - min(x_values)) * 0.05  # 5% 偏移
    offset_y = (max(y_values) - min(y_values)) * 0.05

    # 添加起点箭头和坐标标注
    plt.annotate(f"Start\n({start_x:.2f}, {start_y:.2f})", xy=(start_x, start_y), 
                 xytext=(start_x - offset_x, start_y - offset_y),
                 arrowprops=arrow_params, fontsize=12, color='green')

    # 添加终点箭头和坐标标注
    plt.annotate(f"End\n({end_x:.2f}, {end_y:.2f})", xy=(end_x, end_y), 
                 xytext=(end_x - offset_x, end_y - offset_y),
                 arrowprops=arrow_params, fontsize=12, color='red')

    # 设置标题和坐标轴标签
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # 自动调整坐标轴范围，确保曲线完整显示
    plt.xlim(min(x_values) - offset_x, max(x_values) + offset_x)
    plt.ylim(min(y_values) - offset_y, max(y_values) + offset_y)

    # 保持XY轴比例一致，防止曲线变形
    plt.axis("equal")

    # 显示网格和图例
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 显示图表
    plt.show()