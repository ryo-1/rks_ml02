# -*- coding: utf-8 -*-
import rakus_ml_training as rmt
import pandas as pd
import numpy as np
import classification

if __name__ == '__main__':
    train = rmt.iris.get_train_data()
    test = rmt.iris.get_test_data()

    X = train.drop('target', axis=1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    Y = train['target']
    W = np.ones((1, X.shape[1]))

    W, dmse = classification.fit(W, X, Y)

    X_t = np.hstack((np.ones((test.shape[0], 1)), test))
    Y_t = classification.logistic(X_t, W)

    print(rmt.cancer.confirm(pd.DataFrame(Y_t)))