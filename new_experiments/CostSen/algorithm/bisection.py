import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_fmeasure, get_params_fmeasure

def bisection(X_train, y_train, vec_eta, T, n_class):
    A, B = get_params_fmeasure(n_class)
    C_t = np.ones(shape = (n_class, n_class))*(1/n_class**2)
    alpha, beta = 0, 1

    for t in range(T):
        gamma = (alpha+beta)/2
        L = -(A - gamma*B)
        g_t = get_confusion_matrix_from_loss_no_a(L, X_train, y_train, vec_eta, n_class)
        
        if compute_fmeasure(g_t) >= gamma:
            alpha = gamma
            clf = L
        else:
            beta = gamma

    return clf
