'''
Stanley Bak
Python F-16 GCAS
用于计算侧向阻力系数 Cy
通常用于航空动力学模型以理解飞机在特定操纵面配置下的侧向力。
'''

def cy(beta, ail, rdr):
    'cy function'

    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)
