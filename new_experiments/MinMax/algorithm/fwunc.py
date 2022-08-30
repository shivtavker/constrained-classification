import numpy as np
from standard_funcs.grads import grad_minmax_original
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from scipy.optimize import linprog

def minmaxpostprocess(Cs):
    n = Cs[0].shape[0]
    pi_arr = []

    for i in range(n):
        pi_arr.append(Cs[0][i].sum())

    T = len(Cs)
    ### [v, T-alphas]
    c = np.array([1] + [0]*T)
    A_ub = []

    for i in range(n):
        a_ub_row = [-1]
        for t in range(T):
            a_ub_row.append(-1*Cs[t][i][i]/pi_arr[i])
        
        A_ub.append(a_ub_row)
    
    b_ub = [-1]*(n)

    A_eq = [[0] + [1]*T]
    b_eq = [1]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    return res['x'][1:]

def frank_wolfe_unc(X_train, y_train, vec_eta, T, n_class):
    Cs = [np.ones(shape = (n_class, n_class))*(1/n_class**2)]
    bgammas = []
    clfs = []

    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()

    for t in range(T):
        C_t = Cs[-1]
        L_t = grad_minmax_original(C_t, n_class, p)
        bgamma_t = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class) 
        C_t_new = C_t*(1 - (2/(t+2))) + (2/(t+2))*bgamma_t
        Cs.append(C_t_new)
        bgammas.append(bgamma_t)
        clfs.append(L_t)

    return (clfs, minmaxpostprocess(bgammas))
