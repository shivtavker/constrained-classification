import numpy as np
from scipy.optimize import linprog
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.grads import grad_hmean_original

def frank_wolfe_post(Cs_custom, n_class, p, epsilon):
    T = len(Cs_custom)
    class_weights = (1/T)*np.ones(T)
    good_weights = (1/T)*np.ones(T)
    # print(T)

    ## Making Ax <= b for Coverage also adding alpha sum to 1
    A_ub = np.zeros(shape = (n_class*2, T))
    A_eq = np.ones(T)
    
    for i in range(n_class):
        for t in range(T):
            C = Cs_custom[t]
            A_ub[i][t] = np.sum(C[:, i])
            A_ub[i+n_class][t] = -np.sum(C[:, i])
    
    A_eq = A_eq.reshape(1, T)
    
    b_ub = np.hstack((p + epsilon, epsilon - p))
    b_eq = [1]
    total_errors = 0
    
    for i in range(500):
        grad_alpha = np.zeros(T)
        grad_psi = grad_hmean_original(np.average(Cs_custom, weights=class_weights, axis=0), n_class, p)        
        for t in range(T):
            C = Cs_custom[t]
            grad_alpha[t] = np.sum(np.multiply(grad_psi, C))

        ##Linear Program
        soln = linprog(c= grad_alpha, A_ub = A_ub, A_eq=A_eq, b_ub=b_ub, b_eq=b_eq, bounds=(0, 1), options={'tol':1e-4, 'maxiter': 100})
        class_weights = soln['x']
        if soln['status'] != 0:
            total_errors += 1
        else:
            good_weights = np.copy(class_weights)
        # print(np.sum(good_weights))
        if total_errors > 5:
            return good_weights
        # print(grad_alpha)

    return good_weights