import numpy as np


def KNN_test(X, Y, a, b, num):
    accuracy = 0
    sample_size = len(b)
    for j in range(sample_size):
        distance = np.array([np.sum(np.square(abs(X[i] - a[j]))) for i in range(len(X))])
        print(distance)
        idx = np.argsort(distance)[:num]
        print(idx)
        knear = np.array([Y[i] for i in idx]).sum()
        label = 1 if knear > 0 else -1
        print(knear)
        if label == b[j]:
            accuracy = accuracy + 1
    return accuracy / float(sample_size)
