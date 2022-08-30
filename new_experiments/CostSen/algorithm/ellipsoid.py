import numpy as np
from solvers.fwpost import frank_wolfe_post
from solvers.primal import solve_primal
from standard_funcs.helpers import compute_hmean

def ellipsoid(X_train, y_train, X_test, y_test, T, n_class, rho=0, inner_t=500, eta_xi=0.01):
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    dim_n = n_class
    R_init = 1000
    Ps = [np.identity(dim_n)*R_init]
    x_centers = [np.zeros(dim_n).reshape(dim_n, 1)]
    Cs_train = []
    Cs_test = []
    final_xis = []

    for t in range(T):
        P = Ps[-1]
        x_center = x_centers[-1]
        mu = np.diag(x_center.flatten())

        xi, C_normal, C_train, C_test = solve_primal(X_train, y_train, X_test, y_test, mu, inner_t, eta_xi, p, n_class)
        final_xis.append(xi)
        Cs_train.append(C_train)
        Cs_test.append(C_test)

        grad_vec = -1*(np.diagonal(xi)- np.diagonal(C_normal)).reshape(dim_n, 1)

        norm_const = 1/np.sqrt((np.dot(np.dot(grad_vec.T, P), grad_vec)))
        alpha = norm_const*rho

        norm_grad = grad_vec*norm_const
        x_center_new = x_center - (((1-(dim_n*alpha))/(dim_n+1))*np.dot(P, norm_grad))
        P_mod_part = (2*(1+(dim_n*alpha))/((dim_n+1)*(1+alpha)))*np.matmul(np.dot(P, norm_grad), np.dot(norm_grad.T, P))
        P_new = ((dim_n**2)/(dim_n**2 - 1))*(1 - alpha**2)*(P - P_mod_part)

        x_centers.append(x_center_new)
        Ps.append(P_new)

    final_weights = frank_wolfe_post(Cs_train, n_class, p)

    return (np.average(Cs_train, weights=final_weights, axis=0), np.average(Cs_test, weights=final_weights, axis=0))
