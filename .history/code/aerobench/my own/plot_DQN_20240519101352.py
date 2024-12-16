import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
"""
missile 4500 
    win_rate
        DQN eps=0 y_1
        DQN eps=0.05 y_2
        DQN eps=decreasing y_3
    returns
        DQN eps=0 z_1
        DQN eps=0.05 z_2
        DQN eps=decreasing z_3
missile 4750
    win_rate
        DQN eps=0 y_11
"""
# 设置中文字体
import numpy as np

# 示例数据

y_1=[30.0, 90.0, 100.0, 30.0, 90.0, 90.0, 70.0, 60.0, 50.0, 100.0, 100.0, 100.0, 80.0, 50.0, 0.0, 10.0, 40.0, 90.0, 100.0, 100.0, 100.0, 50.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0]
y_1=np.array([i*0.01 for i in y_1])
y_2=[20.0, 50.0, 10.0, 80.0, 90.0, 70.0, 50.0, 40.0, 30.0, 30.0, 40.0, 60.0, 100.0, 80.0, 30.0, 0.0, 50.0, 40.0, 20.0, 0.0, 60.0, 60.0, 20.0, 40.0, 50.0, 30.0, 80.0, 100.0, 100.0, 30.0, 10.0, 60.0, 60.0, 60.0, 100.0, 90.0, 80.0, 80.0, 40.0, 40.0, 90.0, 100.0, 90.0, 100.0, 40.0, 10.0, 30.0, 20.0, 40.0, 90.0]
y_2=np.array([i*0.01 for i in y_2])
y_3= [20.0, 20.0, 0.0, 10.0, 30.0, 10.0, 0.0, 0.0, 80.0, 90.0, 30.0, 50.0, 50.0, 30.0, 0.0, 40.0, 50.0, 0.0, 20.0, 70.0, 50.0, 70.0, 30.0, 80.0, 40.0, 80.0, 90.0, 60.0, 40.0, 90.0, 50.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0, 40.0, 0.0, 60.0, 100.0, 90.0, 80.0, 100.0, 100.0, 100.0, 60.0, 60.0, 50.0, 30.0]
y_3=np.array([i*0.01 for i in y_3])
y_11=[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 30.0, 50.0, 20.0, 0.0, 0.0, 10.0, 50.0, 0.0, 10.0, 40.0, 30.0, 60.0, 50.0, 20.0, 0.0, 0.0, 0.0, 0.0, 90.0, 10.0, 60.0, 0.0, 50.0, 50.0, 80.0, 10.0, 20.0, 30.0, 50.0, 100.0, 80.0, 70.0, 100.0, 10.0, 0.0, 40.0, 20.0, 50.0, 100.0, 80.0, 100.0, 30.0, 100.0]
y_11=np.array([i*0.01 for i in y_11])
y_12=[20.0, 20.0, 10.0, 10.0, 20.0, 10.0, 0.0, 40.0, 0.0, 0.0, 0.0, 10.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 0.0, 0.0, 0.0, 0.0, 50.0, 60.0, 20.0, 20.0, 30.0, 100.0, 100.0, 80.0, 100.0, 40.0, 100.0, 20.0, 30.0, 30.0, 20.0, 90.0, 100.0, 20.0, 40.0, 100.0, 80.0, 80.0, 100.0, 100.0, 100.0, 100.0]
y_12=np.array([i*0.01 for i in y_12])
y_13= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 20.0, 40.0, 10.0, 30.0, 0.0, 0.0, 60.0, 10.0, 40.0, 100.0, 40.0, 0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 80.0, 40.0, 40.0, 30.0, 100.0, 100.0, 90.0, 100.0, 40.0, 40.0, 100.0, 100.0, 60.0, 30.0, 30.0, 80.0, 100.0, 100.0, 100.0, 100.0, 70.0, 60.0, 50.0] 
y_13=np.array([i*0.01 for i in y_13])
z_1 = np.array( [-4232.192604338592, -1059.8189762735535, -834.4666227386413, -5003.794039611087, -2063.192859898033, -1366.0133843687283, -2628.7930957128624, -2828.8711757369174, -3420.9789343740367, -572.859257334426, -579.7831292358496, -581.506894187927, -1780.9342209609208, -3205.91174284889, -6140.791824075636, -5304.327080631328, -4441.390598642454, -1191.1402159314077, -572.7903245501751, -576.2971789453429, -584.543738635459, -3422.7710482511566, -1717.21787141072, -877.0588753665372, -663.2923156070574, -625.7244907010916, -685.2005486462306, -735.3341439153212, -895.5536881730293, -2654.180429553703, -3269.79025652326, -696.8965860627053, -907.2044802468967, -798.6550069589769, -629.181162947646, -595.0264644135881, -572.7903245501751, -578.2964283255026, -576.4658051266484, -583.8362358783497, -572.7903245501751, -572.7903245501751, -572.7903245501751, -572.7903245501751, -615.7335368563275, -580.227234472708, -3408.8971945523867, -801.7611769224322, -755.9488285160428, -678.7354743342403])
z_2 = np.array([-4735.655397813267, -3165.105202746766, -5188.661917267351, -1621.2418434142116, -1156.3024756811506, -2731.346652822739, -3454.222063694746, -3840.455059687984, -5263.153896257251, -5685.34359728578, -5287.273974713809, -2954.1339936722857, -1040.3627410989527, -1592.8181352756626, -4174.098015798462, -5737.970605013209, -3186.0009073062283, -3779.2282671314374, -5130.129367009727, -6060.330542241172, -4749.15564114118, -4733.027839401579, -5237.863347667228, -4786.2789731968105, -5026.121522512762, -5554.2241553531085, -3967.117928443697, -3474.45843074093, -4816.159977475778, -5570.542505739603, -5232.635596900497, -3921.7862141580176, -3944.5481517500293, -3579.5738586433704, -3612.13560658118, -3874.264649895441, -1638.2219704977344, -2745.213435363975, -4165.625666466551, -4347.084753223229, -3381.6922762430377, -3006.2504906400077, -2918.609045617059, -3181.5847998619874, -5424.692850397335, -5412.213692358259, -4848.303069376445, -5151.936178967878, -4215.4772166904695, -4958.842454712406])
z_3=np.array([-4860.642504663826, -4700.924115780226, -5797.838862106155, -5353.884036783177, -4706.288076454324, -5554.169933318117, -6296.9810197384, -6089.704552266039, -1637.9040097458114, -4796.328297505682, -4470.539020893783, -3074.6608586326756, -3842.300801428047, -4636.650258790574, -6367.959227353523, -5699.873094511911, -3492.6361157390397, -5730.773384532197, -4712.990832604303, -2053.53643855439, -3098.862529508894, -2083.7244141479214, -5990.81138027636, -2680.117241092102, -3587.876338738794, -3750.756961394768, -2411.707697132827, -2568.35951787354, -3575.434681105386, -1098.8130529785246, -3431.2792839492686, -4065.3769531979183, -4050.5451270114136, -3773.3621945400337, -3641.932858924979, -4047.553163314076, -3101.525572174331, -3691.897811921824, -6481.44484200263, -3543.080219752039, -4462.127807902223, -3786.522063411975, -4091.729315617861, -2561.8653851208064, -3115.9345189453875, -4325.05753975424, -5062.382834573691, -5259.131937112883, -4740.285022383741, -5441.656518680493])
z_11=[-5774.459657024984, -5776.781622481511, -5829.382042534321, -5812.138213527156, -5598.7056327278815, -5830.539703140863, -5549.846550851113, -5517.5375481101755, -4541.788944288308, -5758.299755409529, -5695.125171017494, -5551.514178155347, -5794.958133281073, -5063.640361552083, -6084.857613429822, -5628.485563081485, -4966.47066638946, -5574.672147583369, -5254.425683055673, -5246.848666297266, -5202.450256359382, -5668.491597826675, -5637.136388048495, -5655.4747364130735, -5751.540337828121, -5239.724708328254, -5031.586129020843, -5408.709073258409, -5991.862525749044, -5494.855124906664, -5443.146628151471, -5173.45376962962, -5630.0284406840365, -5419.283138237619, -5811.396762896919, -5011.6439146093335, -5242.2991079801095, -5197.847394413733, -4560.7091425438075, -4340.968005229957, -5560.205952081765, -5643.294652318719, -5659.784391114854, -5714.93377029427, -4763.912015180711, -5146.300135644949, -5328.32247927773, -5226.232971317545, -4850.858551853076, -4818.13284568306, -5214.149464858082, -5276.232971317545, -4998.709312823385, -5322.836525052701, -5276.232971317545, -5276.232971317545, -5276.232971317545, -5276.232971317545, -5276.232971317545, 
-5276.232971317545, -5276.232971317545, -5276.232971317545, -5276.232971317545, -5329.398118859953, -5596.081028860772, -4875.817308771231, -5208.432032738116, -5276.232971317545, -5186.687815196798, -5164.107980508414]
z_12=[-5100.13674600664, -5444.531432714284, -5594.284593835598, -5597.976072106323, -5334.6122125008405, -5187.920419472823, -5838.276708239288, -5477.944694013584, -5882.27263711901, -5850.867724795236, -6033.740831180604, -5679.555321895599, -5248.594519919772, -5616.9516811930425, -5883.890362304738, -5997.217348681004, -5793.998520830638, -5925.8111404931215, -5612.716282837153, -5004.875606754503, -5843.920241914049, -5987.0419897619695, 
-5901.356325456929, -5638.4603567766235, -5334.745129814347, -4901.173807662539, -5335.6436850959335, -5503.634244843185, -4543.276758355743, -4610.036001659032, -4541.196098939439, -5498.279583071332, -4775.360806164097, -5558.477728767231, -5171.5329441583435, -5741.0853418353645, -5078.253073872975, -4997.4797122381815, -5409.385340605184, -5182.702860574629, -5233.225078785517, -5620.088435765361, -4834.5560297351185, -5212.829445782682, -4492.3550571451215, -4555.604786663031, -5297.536890007086, -5182.296318920276, -5205.214051024658, -5140.599759058988]
z_13=[-5633.944893271466, -5726.952007818025, -5610.903736002678, -5710.878272688716, -5594.544590191693, -5619.771440329828, -5679.99786176371, -5816.28186051412, -6041.194034600276, -5021.006413307657, -5820.520385646857, -5542.85605882357, -6023.896530156405, -5118.631490299663, -5557.319872425278, -5932.7423061148265, -4614.330750966481, -5584.718640050338, -5596.310150234598, -4707.898274286546, -4900.183052221686, -5642.813554831941, -4856.109544295788, -5632.693875978124, -5648.272393375576, -5597.054114503186, -5592.927027009375, -5220.112549729485, -5468.974473887412, -5256.737746059436, -5614.791647316371, -5183.703177099252, -5187.011178309075, -5219.847721719533, -5175.786429127537, -5765.671045052493, -5134.217410925123, 
-5150.216675574221, -5190.038238771111, -5409.585090314666, -5467.018689016639, -5378.391006688233, -5034.536068904021, -4639.753635977436, -5132.759216956956, -5200.787670015007, -4421.6842390624715, -4385.419358705027, -4550.710183187453, -5082.458228044235]

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
    plt.plot(x, y_1_smooth, marker='o', linestyle='-', color='b', label='eps=0',markersize=2)
    plt.plot(x, y_2_smooth, marker='o', linestyle='-', color='r', label='eps=0.05',markersize=2)
    plt.plot(x, y_3_smooth, marker='o', linestyle='-', color='orange', label='eps decreasing',markersize=2)

    shadow_width_1 = 500  # 阴影宽度
    shadow_width_2=500
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
    plt.title('Return Curve of Different Greedy Degrees of DQN')
    plt.xlabel('Iteration(moving average)')
    plt.ylabel('Returns')
    # 添加网格
    plt.grid(True)
    # 显示图例
    plt.legend(loc='lower right',prop={'size':6})
    # 显示图表
    plt.show()
if __name__=="__main__":
    plot_fig(y_11,y_12,y_13)
