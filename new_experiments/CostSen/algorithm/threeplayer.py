import numpy as np
from standard_funcs.helpers import compute_hmean
from standard_funcs.grads import grad_hmean
from standard_funcs.cost_sensitive import get_confusion_matrix_lr
from standard_funcs.helpers import project_A

def threeplayer(X_train, y_train, X_test, y_test, T, n_class, eta_xi=0.01, eta_mu=0.01):
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    
    mus = [np.ones(shape=(n_class, n_class))]
    clfs = []
    Cs_train = []
    Cs_test = []
    xis = [np.diag(p)]

    for t in range(T):
        mu_t = mus[-1]
        L_t = -1*mu_t
        for i in range(n_class):
            L_t[i, i] = L_t[i, i]/p[i]

        clfs.append(L_t)
        C_train, C_test = get_confusion_matrix_lr(L_t, X_train, y_train, n_class, X_test, y_test)
        Cs_train.append(C_train)
        Cs_test.append(C_test)

        ### Normalize the Matrix
        C_t_normal = np.copy(C_train)
        for i in range(n_class):
            C_t_normal[i, i] = C_train[i, i]/p[i]
        xi_t = xis[-1]

        grad_xi = grad_hmean(xi_t, n_class, p) + mu_t
        grad_xi = grad_xi.clip(-0.5, 0.5)
        xi_t_new = project_A(xi_t - eta_xi*grad_xi)
        mu_t_new = mu_t + eta_mu*(xi_t_new - C_t_normal)
        mus.append(mu_t_new)
        xis.append(xi_t_new)

    return np.average(Cs_train, axis=0), np.average(Cs_test, axis=0)
