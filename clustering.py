import numpy as np


def K_Means(X, n, mu):
    import random

    if (len(mu) == 0):
        randomlist = np.array(random.sample(X, n))
        print(randomlist)
        mu = randomlist
    print(mu)
    clusterSet = [[] for i in range(n)]
    for x in X:
        closest = 999
        cluster = 0
        # print("sample")
        for i in range(n):
            distance = np.sum(np.square(abs(x - mu[i])))

            if distance < closest:
                closest = distance
                cluster = i

        # print(cluster,x,closest)
        clusterSet[cluster].append(x)
    print("0", clusterSet[0])
    print("1", clusterSet[1])
    clus = np.array(clusterSet)
    print(clus)
    newmu = [np.mean(clus[i]) for i in range(n)]
    print(newmu)
    print("nexIteration\n\n")

    if np.array_equal(mu, newmu):
        print("since mu are same so we are done")
        print(newmu)
        return newmu
    else:
        return K_Means(X, n, newmu)
