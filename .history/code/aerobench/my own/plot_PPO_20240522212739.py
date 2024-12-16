"""
missile=4500

    win_rate 
        PPO y_1
        DQN eps=0 y_2
    returns
        PPO z_1
        DQN eps=0 z_2
missile=4750
    win_rate 
        PPO y_11
        DQN eps=0 y_12
    returns
        PPO z_11
        DQN eps=0 z_12
"""
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Microsoft YaHei'
y_1=[50.0, 100.0, 100.0, 100.0, 100.0, 90.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
y_1=np.array([i*0.01 for i in y_1])
y_2=[30.0, 90.0, 100.0, 30.0, 90.0, 90.0, 70.0, 60.0, 50.0, 100.0, 100.0, 100.0, 80.0, 50.0, 0.0, 10.0, 40.0, 90.0, 100.0, 100.0, 100.0, 50.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0]
y_2=np.array([i*0.01 for i in y_2])
y_3=np.ones(y_2.shape)
z_1= [-3733.785583620288, -2843.979118343976, -2899.39143559029, -1769.594530527506, -1325.1590641889259, -1945.7965416527536, -4164.932620432076, -4441.5164244871385, -4190.9626749096515, -6450.230156715537, -6310.645869828253, -6225.936803523235, -6238.676972230312, -6231.987294030613, -6205.394275382176, -6216.174882461497, -6205.658841491959, -6201.038274309561, -6225.86064911008, -6224.8221291117725, -6219.13995829072, -6217.086383933289, 
-6199.161721207272, -6203.208960064079, -6202.54540148158, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6202.652002848705, -6203.150177433867, -6202.119562778158, -6199.161721207272, -6206.562302765711, -6199.776611724343, -6213.774366572948, -6202.481678610374, -6199.161721207272, -6199.161721207272, -6209.6810506033435, -6199.161721207272, -6199.161721207272, -6208.946051452494, -6204.559158306003, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272]
z_2= np.array( [-4232.192604338592, -1059.8189762735535, -834.4666227386413, -5003.794039611087, -2063.192859898033, -1366.0133843687283, -2628.7930957128624, -2828.8711757369174, -3420.9789343740367, -572.859257334426, -579.7831292358496, -581.506894187927, -1780.9342209609208, -3205.91174284889, -6140.791824075636, -5304.327080631328, -4441.390598642454, -1191.1402159314077, -572.7903245501751, -576.2971789453429, -584.543738635459, -3422.7710482511566, -1717.21787141072, -877.0588753665372, -663.2923156070574, -625.7244907010916, -685.2005486462306, -735.3341439153212, -895.5536881730293, -2654.180429553703, -3269.79025652326, -696.8965860627053, -907.2044802468967, -798.6550069589769, -629.181162947646, -595.0264644135881, -572.7903245501751, -578.2964283255026, -576.4658051266484, -583.8362358783497, -572.7903245501751, -572.7903245501751, -572.7903245501751, -572.7903245501751, -615.7335368563275, -580.227234472708, -3408.8971945523867, -801.7611769224322, -755.9488285160428, -678.7354743342403])
y_11=[0.0, 0.0, 0.0, 30.0, 100.0, 90.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 100.0, 90.0, 70.0, 80.0, 70.0, 40.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 100.0, 100.0, 100.0, 100.0]
y_11=np.array([i*0.01 for i in y_11])
y_12=[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 30.0, 50.0, 20.0, 0.0, 0.0, 10.0, 50.0, 0.0, 10.0, 40.0, 30.0, 60.0, 50.0, 20.0, 0.0, 0.0, 0.0, 0.0, 90.0, 10.0, 60.0, 0.0, 50.0, 50.0, 80.0, 10.0, 20.0, 30.0, 50.0, 100.0, 80.0, 70.0, 100.0, 10.0, 0.0, 40.0, 20.0, 50.0, 100.0, 80.0, 100.0, 30.0, 100.0]
y_12=np.array([i*0.01 for i in y_12])
z_11= [-5875.711796487346, -5899.272896331543, -5915.116507267099, -4652.0344702797465, -3002.3846246493995, -3067.355633520377, -3064.747104159674, -1678.9298218951385, -2746.6622379606843, -3439.3143749878423, -2317.1639687194306, -2086.7516127812783, -2605.7659664821504, -2599.3332851543983, -1949.55775480611, -3414.159514123842, -2651.640206271276, -3081.177467823828, -4564.548531352333, -2838.2321355778377, -2000.7660066224676, -1505.6632249434247, -1532.1070361713541, -1780.2886425339207, -2431.871951322835, -2558.938552080418, -1714.5554127106268, -1730.0778890229442, -1680.005085196548, -1469.2491488507417, -1373.2046867773602, -1492.713430751988, -1205.2526575011916, -2087.327899482432, -1305.9587692274774, -1802.5566400740258, -1858.7784791385025, -2173.9819272714394, -2116.4107464449135, -1440.639951384111, -1455.5215833326174, -1776.815314194082, -1393.3829149160974, -1348.5086759466324, -1643.9099553824076, -2691.4279037414335, -2642.9592116011568, -1524.1214793630297, -2425.3024582964927, -1635.3046982735843] 
z_12=[-5774.459657024984, -5776.781622481511, -5829.382042534321, -5812.138213527156, -5598.7056327278815, -5830.539703140863, -5549.846550851113, -5517.5375481101755, -4541.788944288308, -5758.299755409529, -5695.125171017494, -5551.514178155347, -5794.958133281073, -5063.640361552083, -6084.857613429822, -5628.485563081485, -4966.47066638946, -5574.672147583369, -5254.425683055673, -5246.848666297266, -5202.450256359382, -5668.491597826675, -5637.136388048495, -5655.4747364130735, -5751.540337828121, -5239.724708328254, -5031.586129020843, -5408.709073258409, -5991.862525749044, -5494.855124906664, -5443.146628151471, -5173.45376962962, -5630.0284406840365, -5419.283138237619, -5811.396762896919, -5011.6439146093335, -5242.2991079801095, -5197.847394413733, -4560.7091425438075, -4340.968005229957, -5560.205952081765, -5643.294652318719, -5659.784391114854, -5714.93377029427, -4763.912015180711, -5146.300135644949, -5328.32247927773, -5226.232971317545, -4850.858551853076, -4818.13284568306]
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
    plt.plot(x, y_1_smooth, marker='o', linestyle='-', color='b', label='PPO',markersize=0.5)
    plt.plot(x, y_2_smooth, marker='o', linestyle='-', color='r', label='DQN',markersize=0.5)
    # plt.plot(x, y_3_smooth, marker='o', linestyle='-', color='orange', label='eps decreasing',markersize=2)

    # plt.fill_between(x, y_3_upper, y_3_lower, color='orange', alpha=0.1)#alpha是不透明度
    # 添加标题和标签
    plt.title('PPO和DQN的回报曲线')
    plt.xlabel('迭代轮次(滑动平均)')
    plt.ylabel('回报')
    # 添加网格
    plt.grid(True)
    # 显示图例
    plt.legend(loc='lower right',prop={'size':6})
    # 显示图表
    plt.show()
if __name__=="__main__":
    plot_fig(y_11,y_12,y_3)