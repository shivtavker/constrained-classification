import numpy as np
from standard_funcs.grads import grad_hmean_original
from standard_funcs.cost_sensitive import get_confusion_matrix_lr
from standard_funcs.helpers import compute_hmean

def frank_wolfe_unc(X_train, y_train, X_test, y_test, T, n_class):
    Cs = [np.ones(shape = (n_class, n_class))*(1/n_class**2)]
    Cs_test = [np.ones(shape = (n_class, n_class))*(1/n_class**2)]

    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    first = True
    for t in range(T):
        if t > 0:
            first = False
        C_t = Cs[-1]
        L_t = grad_hmean_original(C_t, n_class, p)
        # L_t = 1/abs(np.min(L_t)) * L_t
        bgamma_t, bgamma_t_test = get_confusion_matrix_lr(L_t, X_train, y_train, n_class, X_test, y_test, first) 
        C_t_new = C_t*(1 - (2/(t+2))) + (2/(t+2))*bgamma_t
        C_t_new_test = Cs_test[-1]*(1 - (2/(t+2))) + (2/(t+2))*bgamma_t_test
        Cs.append(C_t_new)
        Cs_test.append(C_t_new_test)
        # print(round(1 - compute_hmean(C_t_new), 2))

    return (Cs[-1], Cs_test[-1])
