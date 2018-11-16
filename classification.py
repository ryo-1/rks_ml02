# -*- coding: utf-8 -*-
import numpy as np


def logistic(x, w):
    y = 1 / (1 + np.exp(-(x.dot(w.T))))
    return y


def dcee_logistic(w, x, t, dcee):
    X_n = x.shape[0]
    y = logistic(x, w)

    dcee = dcee + x.T.dot((y - t.reshape(y.shape[0], 1)))
    dcee = dcee / X_n

    return dcee.T


def ﬁt(w, x, t):
    alpha = 0.1
    i_max = 100000
    eps = 0.0001

    dcee = np.zeros(x.shape[1]).reshape((x.shape[1], 1))

    for i in range(1, i_max):
        dmse = dcee_logistic(w, x, t, dcee)
        w = w - alpha * dmse
        print(np.absolute(dmse.max()))
        if np.absolute(dmse.max()) < eps:  # 終了判定
            break

    return w, dmse


def sigmoid(w, x):
    y = np.exp(-(x.dot(w.T)))
    wk = np.sum(y, axis=1)
    y = y.T / wk
    return y.T
