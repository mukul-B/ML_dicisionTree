import numpy as np
import math


class Node:
    def __init__(self, label, feature):
        self.label = label
        self.feature = feature


class Tree(object):
    "Generic tree node."

    def __init__(self, name, children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.name.label) + "," + repr(self.name.feature) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)


def hypertropy(Y):
    YCount = len(Y)
    unique, counts = np.unique(Y, return_counts=True)
    f = lambda x: x / float(YCount)
    # print(unique, counts )
    probs = f(counts)
    # print(probs)
    lognp = np.log2(probs)
    # print(lognp)
    ho = np.multiply(np.multiply(probs, lognp), -1)
    ho = np.sum(ho)
    # print("ho",ho)
    return ho


def best_feature(X, Y):
    sample_size = len(Y)
    feature_size = len(X[0])
    h0 = hypertropy(Y)
    best_feature = 0
    max_entroy = 0
    leftPart = []
    rightPart = []
    if (h0 == 0):
        return -1
    for i in range(feature_size):
        # print("feature {}".format(i))
        leftD = []
        rightD = []

        for j in range(sample_size):
            if (X[j][i] == 0.0):
                leftD.append(Y[j])

            else:
                rightD.append(Y[j])

            # print(X[j][i], Y[j])
        # print(leftD, len(leftD) / float(sample_size), hypertropy(np.array(leftD)))
        # print(rightD, len(rightD) / float(sample_size), hypertropy(np.array(rightD)))
        ig = h0 - (len(leftD) / float(sample_size)) * hypertropy(np.array(leftD)) - (
                len(rightD) / float(sample_size)) * hypertropy(np.array(rightD))
        if (ig > max_entroy):
            max_entroy = ig
            best_feature = i
            leftPart = leftD
            rightPart = rightD
        # print("ig", ig)
    # print(best_feature, max_entroy)
    return best_feature


def DT_train_binary(X, Y, max_depth):
    leftX = []
    rightX = []
    leftY = []
    rightY = []
    # print(max_depth)

    # print(X)
    if len(X) == 0:
        return
    label = Y.max()
    if max_depth == 0:
        bf = -1
    else:
        bf = best_feature(X, Y)
    t = Tree(Node(label, bf))
    if bf == -1:
        return t
    for i in range(len(Y)):
        if X[i][bf] == 0.0:
            leftX.append(X[i])
            leftY.append(Y[i])
        else:
            rightX.append(X[i])
            rightY.append(Y[i])

    # print("left")
    opL = DT_train_binary(np.array(leftX), np.array(leftY), max_depth - 1)
    # print("right")
    opR = DT_train_binary(np.array(rightX), np.array(rightY), max_depth - 1)

    if (isinstance(opL, Tree)):
        t.add_child(opL)
    if (isinstance(opR, Tree)):
        t.add_child(opR)
    # print(t.name)

    return t


def DT_test_binary(X, Y, DT):
    print("test")
    sample_size = len(Y)
    print(DT)
    accuracy = 0
    for i in range(sample_size):
        T = DT
        while T.name.feature != -1:
            #print(T.name.feature)
            if X[i][T.name.feature] == 0.0:
                T = T.children[0]
            else:
                T = T.children[1]
        if T.name.label == Y[i]:
            accuracy = accuracy + 1
    return accuracy / sample_size
