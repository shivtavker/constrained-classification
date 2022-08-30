import numpy as np
from standard_funcs.helpers import compute_hmean, compute_cov_value
from standard_funcs.grads import grad_hmean, grad_cov
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import project_A, compute_cov_class
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco

def threeplayer(X_train, y_train, vec_eta, T, n_class, epsilon, target, eta_xi=0.01, eta_mu=0.01, eta_lam=0.01):
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    
    mus = [np.ones(shape=(n_class, n_class))]
    coverage_k_array = np.zeros(n_class*2)
    clfs = []
    Cs = []
    xis = [np.diag(p)]
    lams_t = [5]*n_class*2
    scores = []
    cons = []

    for t in range(T):
        mu_t = mus[-1]
        L_t = -1*mu_t
        for i in range(n_class):
            L_t[i, i] = L_t[i, i]/p[i]

        clfs.append(L_t)
        C_t = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class)
        Cs.append(C_t)
        scores.append(compute_hmean(C_t))
        cons.append([compute_cov_value(C_t, p) - epsilon])

        ### Normalize the Matrix
        C_t_normal = np.copy(C_t)
        for i in range(n_class):
            C_t_normal[i, i] = C_t[i, i]/p[i]
        
        xi_t = xis[-1]
        xi_t_unnormalized = np.copy(xi_t)
        for i in range(n_class):
            xi_t_unnormalized[i, i] = xi_t[i, i]*p[i]

        for i in range(n_class):
            coverage_k_array[i], coverage_k_array[i+n_class] = compute_cov_class(xi_t_unnormalized, epsilon, i, target, False), compute_cov_class(xi_t_unnormalized, epsilon, i, target, True)

        grad_xi = grad_hmean(xi_t, n_class, p) + mu_t

        for i in range(n_class):
            grad_cov_lambda_pos = grad_cov(xi_t, n_class, p, i, False)
            grad_cov_lambda_neg = grad_cov(xi_t, n_class, p, i, True)
            
            grad_xi += grad_cov_lambda_pos*lams_t[2*i] + grad_cov_lambda_neg*lams_t[2*i+1]

        grad_xi = grad_xi.clip(-5, 5)

        xi_t_new = project_A(xi_t - eta_xi*grad_xi)
        mu_t_new = mu_t + eta_mu*(xi_t_new - C_t_normal)

        for i in range(n_class):
            lams_t[2*i] = max(lams_t[2*i] + eta_lam*(coverage_k_array[i]), 0)
            lams_t[2*i+1] = max(lams_t[2*i+1] + eta_lam*(coverage_k_array[i+n_class]), 0)

        mus.append(mu_t_new)
        xis.append(xi_t_new)
    
    # cons = np.array(cons)
    # print(np.count_nonzero(cons <= epsilon))

    weights = tfco.find_best_candidate_distribution(np.array(scores), np.array(cons))
    # print(np.count_nonzero(weights > 0))
    # print(sum(weights))

    # weights = (1/T)*np.ones(shape=(T, ))

    return (clfs, weights)
