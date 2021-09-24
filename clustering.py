import numpy as np
import random


def K_Means_r(X, k, mu):
    if len(mu) == 0:
        while True:
            random_list = np.array(random.sample(X, k))
            if len(np.unique(random_list)) != len(random_list):
                mu = random_list
                break
    cluster_set = [[] for i in range(k)]
    for x in X:
        closest = 999
        cluster = 0
        for i in range(k):
            distance = np.sum(np.square(abs(x - mu[i])))
            if distance < closest:
                closest = distance
                cluster = i
        cluster_set[cluster].append(x)
    clus = np.array(cluster_set)
    newmu = [np.mean(clus[i], axis=0) for i in range(k)]
    if np.array_equal(mu, newmu):
        return newmu
    else:
        return K_Means_r(X, k, newmu)


def K_Means(X, k, mu):
    if len(mu) == 0:
        while True:
            random_list = np.array(random.sample(X, k))
            if len(np.unique(random_list)) != len(random_list):
                mu = random_list
                break
    while True:
        cluster_set = [[] for i in range(k)]
        for x in X:
            closest = 999
            cluster = 0
            for i in range(k):
                distance = np.sum(np.square(abs(x - mu[i])))
                if distance < closest:
                    closest = distance
                    cluster = i
            cluster_set[cluster].append(x)
        clusters = np.array(cluster_set)
        new_mu = [np.mean(clusters[i], axis=0) for i in range(k)]
        if np.array_equal(mu, new_mu):
            return new_mu
        else:
            mu = new_mu


def K_Means_better(X, K):
    cluster_set = []
    for i in range(1000):
        clus = K_Means(X, K, [])
        cluster_set.append([[clus[0]], [clus[1]]])
    values, counts = np.unique(cluster_set, return_counts=True, axis=0)
    ind = np.argmax(counts)
    return values[ind]
