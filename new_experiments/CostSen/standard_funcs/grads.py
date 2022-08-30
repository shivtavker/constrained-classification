import numpy as np

def grad_hmean(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1*n_class/(C[j, j]**2 +1e-7)
            else:
                W[j, k] = 0
    return W

def grad_hmean_original(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1*n_class*p[j]/(C[j, j]**2 +1e-7)
            else:
                W[j, k] = 0
    return W