import numpy as np
from standard_funcs.grads import grad_hmean
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import project_A
from solvers.fwpost import frank_wolfe_post

def threeplayer(X_train, y_train, vec_eta, T, n_class, eta_xi=0.01, eta_mu=0.01):
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    
    mus = [0.2*np.eye(n_class)]
    clfs = []
    Cs = []
    xis = [np.eye(n_class)]

    for t in range(T):
        mu_t = mus[-1]
        L_t = -1*mu_t
        for i in range(n_class):
            L_t[i, i] = L_t[i, i]/p[i]

        clfs.append(L_t)
        C_t = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class)
        Cs.append(C_t)

        ### Normalize the Matrix
        C_t_normal = np.copy(C_t)
        for i in range(n_class):
            C_t_normal[i, i] = C_t[i, i]/p[i]
        xi_t = xis[-1]

        grad_xi = grad_hmean(xi_t, n_class, p) + mu_t
        grad_xi = grad_xi.clip(-0.5, 0.5)
        xi_t_new = project_A(xi_t - eta_xi*grad_xi)
        mu_t_new = mu_t + eta_mu*(xi_t_new - C_t_normal)
        mus.append(mu_t_new)
        xis.append(xi_t_new)
    
    weights = (1/T)*np.ones(shape=(T,))

    return (clfs, frank_wolfe_post(Cs, n_class, p))
