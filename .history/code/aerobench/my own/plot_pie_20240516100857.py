import matplotlib.pyplot as plt

# 数据
sizes = [64, 36]
labels = ['run_straight', 'run_left_turn']
colors = ['#ff9999', '#66b3ff']  # 自定义颜色
explode = (0, 0.1)  # 突出显示第二个部分

# 绘制饼图
plt.figure(figsize=(8, 8))  # 设置图形大小（可选）
plt.figure(dpi=250)
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140, shadow=True, textprops={'fontsize': 20})  
plt.axis('equal')  # 保持纵横比相等，使饼图为圆形
plt.title('Distribution of Running Directions', fontsize=16, fontweight='bold')  # 添加标题并调整字体样式
plt.legend(loc='lower right', fontsize=12)  # 添加图例并调整字体大小



plt.show()
