import numpy as np
import math
from scipy.stats import mode

_author_ = 'Zhu Peihao'


def knn(test_num, k):
    """
    :return: accuracy
    """
    # Load array information
    train_images = np.load('train_images.npy')
    train_labels = np.load('train_label.npy')
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_label.npy')
    train_images = np.mat(train_images, float).T
    test_images = np.mat(test_images, float).T
    m, n = np.shape(train_images)
    m1, n1 = np.shape(test_images)
    train_temp = train_images - np.mean(train_images, axis=1) * np.mat(np.ones(n))
    test_temp = test_images - np.mean(train_images, axis=1) * np.mat(np.ones(n1))
    U, Sigma = pca(train_temp)
    K = 0
    temp_dim = 0
    sigma_sum = sum(Sigma)
    # Calculate the trace of a matrix, to select the dimension to be reduced
    for i in range(m):
        temp_dim += Sigma[i]
        K = i + 1
        if temp_dim  >= 0.85 * sigma_sum:
            break

    U_reduce = U[:, 0:K]        # reducing dimension transformation
    train_dim = U_reduce.T * train_temp
    test_dim = U_reduce.T * test_temp
    test_d = test_dim[:, 0:test_num]
    test_l = test_labels[0:test_num]
    category = calculate(train_dim, train_labels, test_d, k)
    accur = cal_accuracy(category, test_l)

    return accur


# PCA dimension reduction function
def pca(train_temp):
    m, n = np.shape(train_temp)
    cov_mat = (train_temp * train_temp.T) / n
    U, Sigma, VT = np.linalg.svd(cov_mat)       # singular value decomposition SVD
    return U, Sigma


# calculate the distance and get the category
def calculate(train_images, train_labels, test_images, k):
    m, n = np.shape(train_images)
    m1, n1 = np.shape(test_images)
    category = np.zeros(n1)
    distance = np.zeros(n)
    for i in range(n1):
        for j in range(n):
            diff = train_images[:, j] - test_images[:, i]
            temp_distance = diff.T * diff
            distance[j] = temp_distance
        num = distance.argsort()[0:k]
        arrange = np.zeros(k)
        for l in range(k):
            arrange[l] = train_labels[num[l]]
        category[i] = mode(arrange)[0][0]
    return category


def cal_accuracy(category, test_labels):
    G = np.zeros(len(test_labels))
    for i in range(len(test_labels)):
        if category[i] == test_labels[i]:
            G[i] = 1
    num = np.where(G == 1)
    accur = len(num[0]) / float(len(test_labels))
    return accur


# Test call
if __name__ == '__main__':
    accuracy = knn(1000, 20)
    print accuracy
