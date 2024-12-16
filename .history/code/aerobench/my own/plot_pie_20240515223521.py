import matplotlib.pyplot as plt

# 数据
sizes = [64,36]
labels = ['run_straight','run_left_turn']

# 绘制饼图
plt.figure(figsize=(6, 6))  # 设置图形大小（可选）
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)  # autopct参数用于显示百分比，startangle参数用于设置起始角度
plt.axis('equal')  # 保持纵横比相等，使饼图为圆形
plt.title('Pie Chart')  # 添加标题

# 显示图形
plt.show()
