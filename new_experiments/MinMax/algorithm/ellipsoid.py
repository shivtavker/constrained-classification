import numpy as np
from solvers.fwpost import frank_wolfe_post
from solvers.primal import solve_primal
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

def ellipsoid(X_train, y_train, vec_eta, T, n_class, rho=0, inner_t=500, eta_xi=0.01):
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    dim_n = n_class
    R_init = 1000
    Ps = [np.identity(dim_n)*R_init]
    x_centers = [np.zeros(dim_n).reshape(dim_n, 1)]
    Cs = []
    clfs = []
    final_xis = []

    for t in range(T):
        P = Ps[-1]
        x_center = x_centers[-1]
        mu = np.diag(x_center.flatten())

        xi, C_normal, clf, C = solve_primal(X_train, y_train, mu, inner_t, eta_xi, vec_eta, p, n_class)
        final_xis.append(xi)
        clfs.append(clf)
        Cs.append(C)
        grad_vec = -1*(np.diagonal(xi)- np.diagonal(C_normal)).reshape(dim_n, 1)

        norm_const = 1/np.sqrt((np.dot(np.dot(grad_vec.T, P), grad_vec)))
        alpha = norm_const*rho

        norm_grad = grad_vec*norm_const
        x_center_new = x_center - (((1-(dim_n*alpha))/(dim_n+1))*np.dot(P, norm_grad))
        P_mod_part = (2*(1+(dim_n*alpha))/((dim_n+1)*(1+alpha)))*np.matmul(np.dot(P, norm_grad), np.dot(norm_grad.T, P))
        P_new = ((dim_n**2)/(dim_n**2 - 1))*(1 - alpha**2)*(P - P_mod_part)

        x_centers.append(x_center_new)
        Ps.append(P_new)
    
    # weights = frank_wolfe_post(Cs, n_class, p)
    # weights = np.ones(T)*(1/T)
    # weights = np.array([0]*(T-1) + [1])
    weights = minmaxpostprocess(Cs)

    return (clfs, weights)
