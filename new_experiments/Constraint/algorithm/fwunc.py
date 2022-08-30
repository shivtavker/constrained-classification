import numpy as np
from standard_funcs.grads import grad_hmean_original
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a

def frank_wolfe_unc(X_train, y_train, vec_eta, T, n_class):
    Cs = [np.ones(shape = (n_class, n_class))*(1/n_class**2)]
    bgammas = []
    clfs = []

    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()

    for t in range(T):
        C_t = Cs[-1]
        L_t = grad_hmean_original(C_t, n_class, p)
        bgamma_t = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class) 
        C_t_new = C_t*(1 - (2/(t+2))) + (2/(t+2))*bgamma_t
        Cs.append(C_t_new)
        bgammas.append(bgamma_t)
        clfs.append(L_t)

    return (clfs, "fwupdate")
