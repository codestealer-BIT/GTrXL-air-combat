import matplotlib.pyplot as plt

# 数据
sizes = [35,49,1,11,4]
labels = ['run_straight', 'run_fall','run_right_turn','run_left_turn','run_rise']
colors = ['#ff9999', '#FFA500','#66b3ff','#FFFF00','#FF0000']  # 自定义颜色  #66b3ff是蓝色 #FFFF00黄
explode = (0, 0.1,0,0,0)  # 突出显示第二个部分

# 绘制饼图
plt.figure(figsize=(8, 8),dpi=150)  # 设置图形大小（可选）
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140, shadow=True, textprops={'fontsize': 20})  
plt.axis('equal')  # 保持纵横比相等，使饼图为圆形
plt.title('Distribution of Running Directions', fontsize=10, fontweight='bold')  # 添加标题并调整字体样式
plt.legend(loc='lower right', fontsize=12)  # 添加图例并调整字体大小



plt.show()
