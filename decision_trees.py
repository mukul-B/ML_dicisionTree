import numpy as np


class Node:
    def __init__(self, label, feature, theta=0.0):
        self.label = label
        self.feature = feature
        self.theta = theta

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
        ret = "\t" * level + repr(self.name.label) + "," + repr(self.name.feature) + "," + repr(self.name.theta) + "\n"
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
    probs = f(counts)
    lognp = np.log2(probs)
    ho = np.multiply(np.multiply(probs, lognp), -1)
    ho = np.sum(ho)
    return ho


def best_feature(X, Y):
    sample_size = len(Y)
    feature_size = len(X[0])
    h0 = hypertropy(Y)
    best_feature = 0
    max_entroy = 0
    if h0 == 0:
        return -1
    for i in range(feature_size):
        leftD = []
        rightD = []

        for j in range(sample_size):
            if X[j][i] == 0.0:
                leftD.append(Y[j])

            else:
                rightD.append(Y[j])

        ig = h0 - (len(leftD) / float(sample_size)) * hypertropy(np.array(leftD)) - (
                len(rightD) / float(sample_size)) * hypertropy(np.array(rightD))
        if ig > max_entroy:
            max_entroy = ig
            best_feature = i
    return best_feature


def DT_train_binary(X, Y, max_depth):
    leftX = []
    rightX = []
    leftY = []
    rightY = []
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
    opL = DT_train_binary(np.array(leftX), np.array(leftY), max_depth - 1)
    opR = DT_train_binary(np.array(rightX), np.array(rightY), max_depth - 1)

    if isinstance(opL, Tree):
        t.add_child(opL)
    if isinstance(opR, Tree):
        t.add_child(opR)
    return t


def DT_test_binary(X, Y, DT):
    print("Testing DT")
    sample_size = len(Y)
    print(DT)
    accuracy = 0
    for i in range(sample_size):
        if DT_make_prediction(X[i], DT) == Y[i]:
            accuracy = accuracy + 1
    return accuracy / float(sample_size)


def DT_make_prediction(x, DT):
    T = DT
    while T.name.feature != -1:
        if x[T.name.feature] == 0.0:
            T = T.children[0]
        else:
            T = T.children[1]
    return T.name.label


# ----------------------------------------------------------------

def partitions(X, Y, i, theta):
    leftD = []
    rightD = []

    for j in range(len(Y)):
        if X[j][i] <= theta:
            leftD.append(Y[j])
        else:
            rightD.append(Y[j])
    return leftD, rightD


def bestTheta(X, Y, f):
    sample_size = len(Y)
    h0 = hypertropy(Y)
    best_theta = 0
    max_entroy = 0
    if h0 == 0:
        return -1
    for j in range(sample_size):
        leftD, rightD = partitions(X, Y, i, X[j][f])
        ig = h0 - (len(leftD) / float(sample_size)) * hypertropy(np.array(leftD)) - (
                len(rightD) / float(sample_size)) * hypertropy(np.array(rightD))
        if ig > max_entroy:
            max_entroy = ig
            best_theta = X[j][f]
    return best_theta, max_entroy


def best_feature_real(X, Y):
    feature_size = len(X[0])
    h0 = hypertropy(Y)
    best_feature = 0
    with_theta = 0
    max_entroy = 0
    if h0 == 0:
        return -1, -1
    for i in range(feature_size):
        theta, ig = bestTheta(X, Y, i)
        if ig > max_entroy:
            max_entroy = ig
            best_feature = i
            with_theta = theta
    return best_feature, with_theta


def DT_train_real(X, Y, max_depth):
    leftX = []
    rightX = []
    leftY = []
    rightY = []
    if len(X) == 0:
        return
    label = Y.max()
    if max_depth == 0:
        bf, theta = -1, -1
    else:
        bf, theta = best_feature_real(X, Y)
    t = Tree(Node(label, bf, theta))
    if bf == -1:
        return t
    for i in range(len(Y)):
        if X[i][bf] <= theta:
            leftX.append(X[i])
            leftY.append(Y[i])
        else:
            rightX.append(X[i])
            rightY.append(Y[i])

    opL = DT_train_real(np.array(leftX), np.array(leftY), max_depth - 1)
    opR = DT_train_real(np.array(rightX), np.array(rightY), max_depth - 1)

    if isinstance(opL, Tree):
        t.add_child(opL)
    if isinstance(opR, Tree):
        t.add_child(opR)
    return t


def DT_test_real(X, Y, DT):
    print("Testing DT")
    sample_size = len(Y)
    print(DT)
    accuracy = 0
    for i in range(sample_size):
        if DT_make_prediction_real(X[i], DT) == Y[i]:
            accuracy = accuracy + 1
    return accuracy / float(sample_size)


def DT_make_prediction_real(x, DT):
    T = DT
    while T.name.feature != -1:
        if x[T.name.feature] <= T.name.theta:
            T = T.children[0]
        else:
            T = T.children[1]
    return T.name.label
