'''
Stanley Bak
Python F-16 GCAS
模拟飞机发动机的输出功率与油门位置之间的关系
'''

def tgear(thtl):
    'tgear function'

    if thtl <= .77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg
