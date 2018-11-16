# -*- coding: utf-8 -*-
import numpy as np


def logistic(x, w):
    y = 1 / (1 + np.exp(-(x.dot(w.T))))
    return y


def cee_logistic(w, x, t):
    X_n = x.shape[0]
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t.loc[n] * np.log(y[n]) +
                     (1 - t.loc[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee


def dcee_logistic(w, x, t, dcee):
    X_n = x.shape[0]
    y = logistic(x, w)

    dcee = dcee + x.T.dot((y - t.reshape(y.shape[0], 1)))
    dcee = dcee / X_n

    return dcee.T


def ﬁt(w, x, t):
    alpha = 0.1  # 学習率
    i_max = 100000  # 繰り返しの最大数
    eps = 0.0001  # 繰り返しをやめる勾配の絶対値のしきい値

    dcee = np.zeros(x.shape[1]).reshape((x.shape[1], 1))

    for i in range(1, i_max):
        dmse = dcee_logistic(w, x, t, dcee)
        w = w - alpha * dmse
        print(np.absolute(dmse.max()))
        if np.absolute(dmse.max()) < eps:  # 終了判定
            break

    return w, dmse