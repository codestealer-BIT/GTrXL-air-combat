'''
Stanley Bak
Python F-16

Rtau function

用于根据输入dp（一个表示差异的参数）计算时间常数rt
'''

def rtau(dp):
    'rtau function'

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = .1
    else:
        rt = 1.9 - .036 * dp

    return rt
