import numpy as np

def grad_hmean(C, n_class, p):
    W = np.zeros((n_class, n_class))
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                W[j, k] = -1*n_class/(C[j, j]**2 +1e-5)
            else:
                W[j, k] = 0
    return W

def grad_hmean_vec(Cvec, n_class, p):
    W = np.zeros(2*n_class)
    for i in range(n_class):
        if(i < n_class):
            W[i] = -1*n_class/(Cvec[i]**2 +1e-5)
            
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

def grad_cov(C, n_class, p, k, neg_param = False):
    grad_k = np.zeros(shape = (n_class, n_class))
    for i in range(n_class):
        grad_k[i, k] = 1
    grad_k[k, k] = 1/p[k]
    
    if(neg_param):
        grad_k = -1*grad_k
        
    return grad_k


def grad_CM(u, v, w, lambda_val, n_class, p):
    # new_mat = np.diag(w)
    # for i in range(n_class):
    #     new_mat[i, i] = new_mat[i, i]/p[i]
    return grad_hmean_vec(u, n_class, p) + w + lambda_val*(u-v)

def grad_FAIR(u, v, w, lambda_val, n_class, p):
    # new_mat = np.diag(w)
    # for i in range(n_class):
    #     new_mat[i, i] = new_mat[i, i]/p[i]
    return grad_hmean_vec(v, n_class, p) - w + lambda_val*(v-u)