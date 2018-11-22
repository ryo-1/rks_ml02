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
        if np.absolute(dmse.max()) < eps:
            break

    return w, dmse


def softmax(w, x):
    y = np.exp(w.dot(x.T))
    wk = np.sum(y, axis=1)
    y = y.T / wk
    return y


def cee_softmax(w, x, t):
    X_n = x.shape[0]
    y = softmax(w, x)

    cee = t * np.log(y)
    cee = np.sum(cee, axis=1)
    cee = cee.sum()
    cee = -1 * cee / X_n
    return cee


def dcee_softmax(w, x, t, dcee):
    X_n = x.shape[0]
    y = softmax(w, x)

    dcee = dcee - np.array(x.T.dot((np.array(t - y)))).T
    dcee = dcee / X_n

    return dcee


def ﬁt_softmax(w, x, t):
    alpha = 0.01
    i_max = 1000
    eps = 1

    dcee = np.zeros((w.shape[0], w.shape[1]))

    for i in range(1, i_max):
        dmse = dcee_softmax(w, x, t, dcee)
        w = w - alpha * dmse

        cee = cee_softmax(w, x, t)
        print(i, cee)
        if cee < eps:
            break

    return w, dmse


class score:
    def __init__(self):
        self.xmean = {}
        self.xstd = {}

    def zscore(self, x, key):
        self.xmean[key] = x.mean()
        self.xstd[key] = np.std(x)

        return (x - self.xmean[key]) / self.xstd[key]

    def test_zscore(self, x, key):
        return (x - self.xmean[key]) / self.xstd[key]
