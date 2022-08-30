import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.grads import grad_minmax
from standard_funcs.helpers import project_A

def solve_primal(X_train, y_train, mu, iterations, eta, vec_eta, p, n_class):
    xi_t = np.diag(p)
    L_t = -1*np.copy(mu)
    for i in range(n_class):
        L_t[i, i] = L_t[i, i]/p[i]
    C = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class)
    clf = L_t
    C_normal = np.copy(C)
    
    for i in range(n_class):
        C_normal[i, i] = C[i, i]/p[i]
    
    for t in range(iterations):
        grad_xi = grad_minmax(xi_t, n_class, p) + mu
        xi_t_new = project_A(xi_t - eta*grad_xi)
        xi_t = xi_t_new
        
    return (xi_t_new, C_normal, clf, C)