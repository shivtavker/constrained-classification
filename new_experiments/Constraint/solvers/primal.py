import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.grads import grad_hmean_vec, grad_cov
from standard_funcs.helpers import project_A, compute_cov_class

def solve_primal(X_train, y_train, mu, iterations, eta, vec_eta, p, target, epsilon, lams_t, n_class):
    ## xi as vector of dim 2*n
    xi_t = np.concatenate([p, p], axis=0)
    L_t = []

    for i in range(n_class):
        L_t.append(list(-1*mu[n_class:]))
    
    L_t = np.array(L_t)

    for i in range(n_class):
        L_t[i, i] = -mu[i]/p[i]

    C = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class)
    clf = L_t
    coverage_k_array = np.zeros(n_class*2)
    
    # change_xi_norm = 10

    # while change_xi_norm > 1e-5:
    for _ in range(iterations):
        grad_xi = grad_hmean_vec(xi_t, n_class, p) + mu
        
        for i in range(n_class, 2*n_class):
            grad_xi_curr = np.zeros(2*n_class)
            grad_xi_curr[i] = 1
            grad_xi_curr[i-n_class] = p[i-n_class]
            # grad_xi_curr[i-n_class] = 1

            grad_xi += grad_xi_curr*lams_t[2*(i-n_class)] - grad_xi_curr*lams_t[2*(i-n_class)+1]
        
        xi_t_new = project_A(xi_t - eta*grad_xi)
        # change_xi_norm = np.linalg.norm(xi_t_new - xi_t)
        xi_t = xi_t_new

    # print(change_xi_norm)
    
    # print("Done Inner Optimization")
        
    return (xi_t_new, clf, C)