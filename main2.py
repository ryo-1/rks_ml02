# -*- coding: utf-8 -*-
import rakus_ml_training as rmt
import pandas as pd
import numpy as np
import classification

if __name__ == '__main__':
    func = classification.score()
    train = rmt.iris.get_train_data()
    test = rmt.iris.get_test_data()

    X = train.drop(['target', 'petal length (cm)'], axis=1)
    for key in X.keys():
        X[key] = func.zscore(X[key], key)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    Y = pd.get_dummies(train.target)

    K = len(train.target.value_counts())
    W = np.zeros((K, X.shape[1]))

    W, dmse = classification.ﬁt_softmax(W, X, Y)

    test = test.drop(['petal length (cm)'], axis=1)
    for key in test.keys():
        test[key] = func.test_zscore(test[key], key)
    test = np.hstack((np.ones((test.shape[0], 1)), test))
    test_Y = classification.softmax(W, test)

    print(rmt.iris.confirm(pd.DataFrame(np.array(test_Y).argmax(axis=1))))
