"""
missile=4500
    win_rate 
        PPO y_1
        DQN eps=0 y_2
        GTtXL y_3
    returns
        PPO z_1
        DQN eps=0 z_2
        GTrXL z_3
missile=4750
    win_rate 
        PPO y_11
        DQN eps=0 y_12
        GTrXL y_13
    returns
        PPO z_11
        DQN eps=0 z_12
        GTrXL z_13
missile=4800
    win_rate 
        PPO y_21
        DQN eps=0 y_22
        GTrXL y_23
    returns
        PPO z_21
        DQN eps=0 z_22
        GTrXL z_23
"""
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Microsoft YaHei'
y_1=[50.0, 100.0, 100.0, 100.0, 100.0, 90.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
y_1=np.array([i*0.01 for i in y_1])
y_2=[30.0, 90.0, 100.0, 30.0, 90.0, 90.0, 70.0, 60.0, 50.0, 100.0, 100.0, 100.0, 80.0, 50.0, 0.0, 10.0, 40.0, 90.0, 100.0, 100.0, 100.0, 50.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0]
y_2=np.array([i*0.01 for i in y_2])
y_3=[0.8, 0.5, 0.3, 0.2, 0.2, 0.5, 0.4, 0.3, 0.2, 0.4, 0.1, 0.4, 0.6, 0.6, 0.4, 0.5, 0.3, 0.2, 0.2, 0.4, 0.6, 0.7, 0.2, 0.3, 0.2, 0.0, 0.2, 0.3, 0.4, 0.0, 0.2, 0.3, 0.4, 0.6, 0.5, 0.4, 0.5, 0.1, 0.1, 0.3, 0.4, 0.3, 0.5, 0.3, 0.7, 0.8, 0.9, 0.9, 0.9, 1.0]
z_1= [-3733.785583620288, -2843.979118343976, -2899.39143559029, -1769.594530527506, -1325.1590641889259, -1945.7965416527536, -4164.932620432076, -4441.5164244871385, -4190.9626749096515, -6450.230156715537, -6310.645869828253, -6225.936803523235, -6238.676972230312, -6231.987294030613, -6205.394275382176, -6216.174882461497, -6205.658841491959, -6201.038274309561, -6225.86064911008, -6224.8221291117725, -6219.13995829072, -6217.086383933289, 
-6199.161721207272, -6203.208960064079, -6202.54540148158, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6202.652002848705, -6203.150177433867, -6202.119562778158, -6199.161721207272, -6206.562302765711, -6199.776611724343, -6213.774366572948, -6202.481678610374, -6199.161721207272, -6199.161721207272, -6209.6810506033435, -6199.161721207272, -6199.161721207272, -6208.946051452494, -6204.559158306003, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272, -6199.161721207272]
z_2= np.array( [-4232.192604338592, -1059.8189762735535, -834.4666227386413, -5003.794039611087, -2063.192859898033, -1366.0133843687283, -2628.7930957128624, -2828.8711757369174, -3420.9789343740367, -572.859257334426, -579.7831292358496, -581.506894187927, -1780.9342209609208, -3205.91174284889, -6140.791824075636, -5304.327080631328, -4441.390598642454, -1191.1402159314077, -572.7903245501751, -576.2971789453429, -584.543738635459, -3422.7710482511566, -1717.21787141072, -877.0588753665372, -663.2923156070574, -625.7244907010916, -685.2005486462306, -735.3341439153212, -895.5536881730293, -2654.180429553703, -3269.79025652326, -696.8965860627053, -907.2044802468967, -798.6550069589769, -629.181162947646, -595.0264644135881, -572.7903245501751, -578.2964283255026, -576.4658051266484, -583.8362358783497, -572.7903245501751, -572.7903245501751, -572.7903245501751, -572.7903245501751, -615.7335368563275, -580.227234472708, -3408.8971945523867, -801.7611769224322, -755.9488285160428, -678.7354743342403])
z_3=[1227.1520615865652, -1627.2185665064437, -3474.077500199378, -4567.489537119237, -4587.469974247366, -2067.2178482839436, -2907.075190299584, -3668.276676341975, -4614.011544511572, -2707.1072285612436, -5518.293916926488, -2724.0009462491043, -1000.3429819411028, -609.4830226742681, -2628.124768955374, -1788.7739863073468, -3974.8570301081572, -4615.228173262533, -4818.137615228677, -2625.1836118763867, -651.6442559039926, 170.7805622868183, -4759.033323318551, -3609.913761285542, -4557.240299754886, -6550.475490655555, -4376.879044030551, -3737.1122804686383, -2645.7343741892028, -6362.351038759185, -4546.928671723838, -3587.2225157463827, -3769.95102982781, -581.3157444645765, -1440.367852267966, -2558.826466141746, -1623.2514889713477, -5769.456633482946, -5702.881819834984, -3839.1748350199728, -2799.0801169900164, -3478.207249014954, -1455.975176466639, -3575.0793079433656, -596.6317588808307, 1003.2024585637831, 1289.6845735317315, 2080.6061948331403, 2280.8408821091307, 3374.222634707251]   
y_11=[0.0, 0.0, 0.0, 30.0, 100.0, 90.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 100.0, 90.0, 70.0, 80.0, 70.0, 40.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 100.0, 100.0, 100.0, 100.0]
y_11=np.array([i*0.01 for i in y_11])
y_12=[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 30.0, 50.0, 20.0, 0.0, 0.0, 10.0, 50.0, 0.0, 10.0, 40.0, 30.0, 60.0, 50.0, 20.0, 0.0, 0.0, 0.0, 0.0, 90.0, 10.0, 60.0, 0.0, 50.0, 50.0, 80.0, 10.0, 20.0, 30.0, 50.0, 100.0, 80.0, 70.0, 100.0, 10.0, 0.0, 40.0, 20.0, 50.0, 100.0, 80.0, 100.0, 30.0, 100.0]
y_12=np.array([i*0.01 for i in y_12])
y_13=[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 1.0, 0.9, 0.9, 0.9, 0.8, 0.9, 0.8, 1.0, 1.0, 1.0, 0.8, 0.9, 0.9, 1.0, 0.8, 0.8, 0.8, 0.4, 0.7, 0.8, 0.7, 0.6, 0.7, 0.6, 0.8, 0.6, 0.7, 0.5, 0.7, 0.2, 0.2, 0.2, 0.0, 0.0, 0.5, 0.0, 0.1]
z_11= [-5875.711796487346, -5899.272896331543, -5915.116507267099, -4652.0344702797465, -3002.3846246493995, -3067.355633520377, -3064.747104159674, -1678.9298218951385, -2746.6622379606843, -3439.3143749878423, -2317.1639687194306, -2086.7516127812783, -2605.7659664821504, -2599.3332851543983, -1949.55775480611, -3414.159514123842, -2651.640206271276, -3081.177467823828, -4564.548531352333, -2838.2321355778377, -2000.7660066224676, -1505.6632249434247, -1532.1070361713541, -1780.2886425339207, -2431.871951322835, -2558.938552080418, -1714.5554127106268, -1730.0778890229442, -1680.005085196548, -1469.2491488507417, -1373.2046867773602, -1492.713430751988, -1205.2526575011916, -2087.327899482432, -1305.9587692274774, -1802.5566400740258, -1858.7784791385025, -2173.9819272714394, -2116.4107464449135, -1440.639951384111, -1455.5215833326174, -1776.815314194082, -1393.3829149160974, -1348.5086759466324, -1643.9099553824076, -2691.4279037414335, -2642.9592116011568, -1524.1214793630297, -2425.3024582964927, -1635.3046982735843] 
z_12=[-5774.459657024984, -5776.781622481511, -5829.382042534321, -5812.138213527156, -5598.7056327278815, -5830.539703140863, -5549.846550851113, -5517.5375481101755, -4541.788944288308, -5758.299755409529, -5695.125171017494, -5551.514178155347, -5794.958133281073, -5063.640361552083, -6084.857613429822, -5628.485563081485, -4966.47066638946, -5574.672147583369, -5254.425683055673, -5246.848666297266, -5202.450256359382, -5668.491597826675, -5637.136388048495, -5655.4747364130735, -5751.540337828121, -5239.724708328254, -5031.586129020843, -5408.709073258409, -5991.862525749044, -5494.855124906664, -5443.146628151471, -5173.45376962962, -5630.0284406840365, -5419.283138237619, -5811.396762896919, -5011.6439146093335, -5242.2991079801095, -5197.847394413733, -4560.7091425438075, -4340.968005229957, -5560.205952081765, -5643.294652318719, -5659.784391114854, -5714.93377029427, -4763.912015180711, -5146.300135644949, -5328.32247927773, -5226.232971317545, -4850.858551853076, -4818.13284568306]
z_13=[697.3201214640691, -6049.32713835738, -5891.052219209001, -6016.1409486025495, -5973.704635877615, -6023.182808644022, -5967.05478847482, -6178.707379602274, -6109.850187077504, -6210.445704285622, -6508.789747859887, -2686.389032952429, -612.0442343226086, 1422.5770251342758, 356.24276050725547, 397.42039522566483, 676.1319606355146, 126.87803222596767, 215.52876717451235, -379.45582124083086, 1046.1259545987589, 790.1312798517804, 1097.065687900264, -43.03996640011319, 318.8343631482894, 163.88990081117535, 1356.9324128093538, -670.2917394484713, -258.0413773018812, -180.48470129958497, -2986.9967198912955, -551.9484236794375, 1075.4740003540865, -79.29063657156476, -1083.446948161193, 22.98438423449834, -354.3325203914954, 1498.04216192424, -440.0202106487112, 206.772227173854, -2071.8075203924773, 69.0702971929245, -4465.022030047932, -4338.753723516986, -4321.82524211157, -6254.874158729696, -6353.2214245076775, -1775.6873014735015, -6128.367800298588, -5349.914029275334]
y_23=[0.7, 0.0, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.3, 0.5, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3, 0.2, 0.3, 0.4, 0.9, 0.9, 0.8, 1.0, 0.7, 0.9, 0.9, 1.0, 1.0, 0.8, 0.9, 0.8, 0.5, 0.4, 0.9, 0.9, 0.9, 1.0, 1.0, 0.2, 0.4, 0.4, 0.7, 0.8, 0.8, 0.7, 0.8, 0.7, 0.7, 1.0, 1.0, 0.7, 0.8, 0.9, 1.0]

y_21=[0.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,100.0,100.0,100.0,100.0,100.0,100.0]
y_21=np.array([i*0.01 for i in y_21])
y_22=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0, 0.0, 20.0, 0.0, 30.0, 30.0, 0.0, 0.0, 20.0, 20.0, 0.0, 50.0, 10.0, 50.0, 30.0, 0.0, 60.0, 0.0, 20.0, 50.0, 0.0, 10.0, 0.0, 70.0, 70.0, 100.0, 20.0, 0.0, 90.0, 100.0, 70.0, 100.0, 100.0, 80.0, 90.0, 100.0, 70.0, 100.0, 100.0, 100.0, 100.0, 90.0, 0.0, 80.0, 100.0, 100.0, 100.0, 100.0]
y_22=np.array([i*0.01 for i in y_22])
z_21= [-5840.82049136529, -4109.8242971778955, -4227.398295239374, -5240.324565902794, -5321.344495303147, -5330.840846063196, -5330.828219876279, -5331.277846059826, -5330.839816449722, -5330.840846063196, -5319.2185927649025, -5331.173067365459, -5330.840846063196, -5314.607626651638, -5330.840846063196, -5330.842982007059, -5330.840846063196, -5330.840846063196, -5330.840846063196, -5330.840846063196, -5310.840846063196, -5330.840846063196, -5330.840846063196, -5330.840846063196, -5330.901396687665, -5330.840846063196, -5318.103139442569, -5330.840846063196, -5330.845270356405, -5331.081884874722, -5330.840846063196, -5318.796517539917, -5330.627923999611, -5330.840846063196, -5316.449991275913, -5330.840846063196, -5330.906279372763, -5330.622682069112, -5331.899182555954, -5319.78719925903, -5326.972765528519, -5313.999990278098, -5330.840846063196, -5330.840846063196, -5333.901007056466, -5330.840846063196, -5330.840846063196, -5330.840846063196, -5319.78719925903, -5330.840846063196,-5330.840846063196,-5330.840846063196,-5330.840846063196,-5330.840846063196,-5330.840846063196,-5330.840846063196]
z_22= [-5716.849499902889, -5655.453726493744, -5825.395084459783, -5801.697150694619, -5706.061294492429, -5554.880235085572, -5667.175366556328, -5152.933167359979, -5202.063618345382, -5495.039167388446, -5673.583746502673, -5651.997331622349, -5779.087125845125, -5600.037613784736, -5495.039167388446, -5721.234520932473, -5764.750872591105, -5213.221536037036, -5943.127385542231, -5298.457793603191, -5443.766225010867, -5187.885957949037, -5104.728404188432, -5854.451343672523, -4638.184822747369, -5942.59813452382, -5367.207560389464, -5156.214819611933, -5744.468682845466, -5462.5350767914315, -5543.3353290174455, -5471.9549836170545, -5273.349356427061, -5330.840846063196, -5488.147477129236, -5510.355626673872, -5120.215572529334, -5051.2418704990105, -5107.1918576792705, -5330.840846063196, -5330.840846063196, -4815.036271477946, -4523.900399133539, -4997.484449989962, -5024.865585683465, -4722.61266797759, -5166.174941424609, -5351.20684354036, -5330.840846063196, -5176.292302983791, -5642.006941823546, -4988.125442613481, -4253.895703223797, -4581.83030621013, -4846.30977897227, -4544.989731659245]
z_23=[-887.1026435076886, -6058.368225964975, -5042.838953008353, -4422.4221191144725, -5239.465030995479, -5138.847058154462, -6107.712594411751, -6109.265770138204, -5205.468834349297, -3743.58282192512, -1702.0168364743993, -5244.757453269934, -6050.977946817144, -3278.0582809631815, -6165.723009250147, -6166.975156994649, -6211.364889339581, -3456.2261427234944, -4250.269595244598, -3433.0904821303584, -2721.781769755808, 632.0969881810121, 
312.2013915340957, -183.08130866758856, 1527.1908717467916, -1061.9069394638195, 1151.141198227763, 677.4979257176765, 1603.6210828638636, 1288.9182719245443, -442.79666768919424, 1532.701577703117, 958.0699167796731, -2162.464699134153, -2681.25382395827, 2154.518763489143, 2271.420058108463, 1882.21065875508, 3615.1538641989014, 3614.074063748068, -4403.417582800941, -2501.6380804118457, -2625.8891393533754, -134.08292686730616, 1026.7066700552118, 1568.8135120583106, 491.8477638722826, 1163.3863812099567, 453.96458960487433, 536.8195369479793, 3341.3732061242117, 3533.9633609793045, 422.819722052728, 1526.7595013944142, 2608.2489715432343, 3779.8915272658555]
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
    plt.plot(x, y_3_smooth, marker='o', linestyle='-', color='orange', label='PPO-GTrXL',markersize=0.5)

    # 添加标题和标签
    plt.title('PPO,DQN和PPO-GTrXL的回报曲线')
    plt.xlabel('迭代轮次(滑动平均)')
    plt.ylabel('回报')
    # 添加网格
    plt.grid(True)
    # 显示图例
    plt.legend(loc='lower right',prop={'size':6})
    # 显示图表
    plt.show()
if __name__=="__main__":
    plot_fig(z_11,z_12,z_13)