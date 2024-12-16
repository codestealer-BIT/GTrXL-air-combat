import matplotlib.pyplot as plt
"""
missile 4500 
DQN eps=0 y_1
DQN eps=0.05 y_2
"""
x=list(range(1,51))
y_1=[0.2,0.3,0.4,0.5,0.2,0,0.6,0.2,0.8]
y_1.extend([1]*41)
y_2=[2,2,0,2,0,3,2,3,3,2,0,3,6,6,1,1,1,5,2,0,0,3,3,4,7,9,9,8,6,4,6,8,7,4,2,3,1,1,0,2,5,0,1,2,3,8,4,0,3,2]*0.1
# 示例数据
x = [1, 2, 3, 4, 5]  # 横坐标数据
y = [2, 3, 5, 7, 11]  # 纵坐标数据

# 创建折线图
plt.plot(x, y, marker='o', linestyle='-', color='b', label='实验结果')

# 添加标题和标签
plt.title('实验结果折线图')
plt.xlabel('实验次数')
plt.ylabel('结果值')

# 添加网格
plt.grid(True)

# 显示图例
plt.legend()

# 显示图表
plt.show()

