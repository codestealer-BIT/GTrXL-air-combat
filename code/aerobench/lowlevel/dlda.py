'''
Stanley Bak
F-16 GCAS in Python
dlda function
用于计算由于副翼偏转引起的侧向力系数的变化
通常用 δa（副翼的偏转角）表示
'''

import numpy as np
from aerobench.util import fix, sign

def dlda(alpha, beta):
    'dlda function'

    a = np.array([[-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017], \
    [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016], \
    [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014], \
    [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012], \
    [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011], \
    [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010], \
    [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k-1, m-1]
    u = a[k-1, n-1]
    v = t + abs(da) * (a[l-1, m-1] - t)
    w = u + abs(da) * (a[l-1, n-1] - u)

    return v + (w - v) * abs(db)
