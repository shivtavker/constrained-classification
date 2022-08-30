import numpy as np
from standard_funcs.cost_sensitive import get_confusion_matrix_lr
from standard_funcs.grads import grad_hmean
from standard_funcs.helpers import project_A

def solve_primal(X_train, y_train, X_test, y_test, mu, iterations, eta, p, n_class):
    xi_t = np.diag(p)
    L_t = -1*np.copy(mu)
    for i in range(n_class):
        L_t[i, i] = L_t[i, i]/p[i]
    C_train, C_test = get_confusion_matrix_lr(L_t, X_train, y_train, n_class, X_test, y_test)
    C_normal = np.copy(C_train)
    
    for i in range(n_class):
        C_normal[i, i] = C_train[i, i]/p[i]
    
    for t in range(iterations):
        grad_xi = grad_hmean(xi_t, n_class, p) + mu
        xi_t_new = project_A(xi_t - eta*grad_xi)
        xi_t = xi_t_new
        
    return (xi_t_new, C_normal, C_train, C_test)