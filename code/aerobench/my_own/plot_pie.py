import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 数据
sizes = [1,99]
# labels = ['定常飞行', '向下俯冲','右转弯','左转弯','向上拉起']
# colors = ['#ff9999', '#FFA500','#66b3ff','#FFFF00','#FF0000']  # 自定义颜色  #66b3ff是蓝色 #FFFF00黄
labels=['定常飞行','向下俯冲']
colors=['#FFA500','#ff9999']
explode = (0, 0.1)  # 突出显示第二个部分

# 绘制饼图
plt.figure(figsize=(8, 8),dpi=150)  # 设置图形大小（可选）
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140, shadow=True, textprops={'fontsize': 25})  
plt.axis('equal')  # 保持纵横比相等，使饼图为圆形
plt.title('动作分布', fontsize=30, fontweight='bold')  # 添加标题并调整字体样式
plt.legend(loc='lower right', fontsize=12)  # 添加图例并调整字体大小



plt.show()
