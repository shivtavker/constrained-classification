import numpy as np
from standard_funcs.helpers import compute_gmean

def grad_hmean(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1*n_class/(C[j, j]**2 +1e-5)
            else:
                W[j, k] = 0
    return W

def grad_minmax(C, n_class, p):
    arr = []
    for i in range(n_class):
        arr.append(1 - C[i][i])

    index = np.argmax(arr)

    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k == index:
                W[j, k] = -1
            else:
                W[j, k] = 0
    return W

def grad_gmean(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1*(1-compute_gmean(C))/(C[j, k]*p[j] + 1e-5)
            else:
                W[j, k] = 0
    return W

def grad_qmean(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = 0
            else:
                W[j, k] = 0
    return W

def grad_hmean_original(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1*n_class*p[j]/(C[j, j]**2 +1e-5)
            else:
                W[j, k] = 0
    return W

def grad_minmax_original(C, n_class, p):
    arr = []
    for i in range(n_class):
        arr.append(1 - (C[i][i]/C[i, :].sum()))

    index = np.argmax(arr)

    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k == index:
                W[j, k] = -1/p[index]
            else:
                W[j, k] = 0
    return W

def grad_qmean_original(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = ((C[j, j]/p[j]) - 1 + 1e-5)/p[j]
            else:
                W[j, k] = 0
    return W

def grad_gmean_original(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1/(C[j, k] + 1e-5)
            else:
                W[j, k] = 0
    return W