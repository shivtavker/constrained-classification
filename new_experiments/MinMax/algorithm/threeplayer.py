import numpy as np
from standard_funcs.grads import grad_minmax
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import project_A
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

        grad_xi = grad_minmax(xi_t, n_class, p) + mu_t
        grad_xi = grad_xi.clip(-0.5, 0.5)
        xi_t_new = project_A(xi_t - eta_xi*grad_xi)
        mu_t_new = mu_t + eta_mu*(xi_t_new - C_t_normal)
        mus.append(mu_t_new)
        xis.append(xi_t_new)
    
    weights = (1/T)*np.ones(shape=(T,))

    return (clfs, minmaxpostprocess(Cs))
