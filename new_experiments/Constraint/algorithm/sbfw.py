import numpy as np
from standard_funcs.grads import grad_hmean_original
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a, get_confusion_vector
from standard_funcs.grads import grad_CM, grad_FAIR
from solvers.lmo_cm import LMO_CM
from solvers.lmo_fair import LMO_FAIR
from standard_funcs.helpers import compute_cov_value, compute_hmean
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco

def split_frank_wolfe(X_train, y_train, vec_eta, n_class, epsilon, target, lambda_val, eta_t_array, T):
    L = np.ones(shape = (n_class, n_class))
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()

    start_cm = get_confusion_matrix_from_loss_no_a(L, X_train, y_train, vec_eta, n_class)
    start_cv = get_confusion_vector(start_cm)
    start_fair = np.concatenate((target, np.zeros(n_class)), axis=0)
    start_clf = L
    start_w = 0.5*np.ones(shape=(2*n_class,))

    u_array = [start_cv]
    v_array = [start_fair]
    w_array = [start_w]
    clfs = [start_clf]
    LMOCs = [start_cv]

    for t in range(T):
        eta_t = eta_t_array[int(t*len(eta_t_array)/T)]
        u_t, v_t, w_t = u_array[-1], v_array[-1], w_array[-1]
        gamma = 2/(t+2) 
        u_new, v_new, w_new, clf, tilde_u = frank_wolfe_update(u_t, v_t, w_t, lambda_val, gamma, eta_t, n_class, p, X_train, y_train, vec_eta, epsilon, target)
        u_array.append(u_new)
        v_array.append(v_new)
        w_array.append(w_new)
        clfs.append(clf)
        LMOCs.append(tilde_u)

    final_u, final_v = u_array[-1], v_array[-1]
    # print(np.linalg.norm(final_u - final_v))
    # print(compute_cov_value(final_v, p))

    Cs = []
    scores = []
    cons = []

    for clf in clfs:
        C = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, n_class)
        score = compute_hmean(C)
        con = max(compute_cov_value(C, target) - epsilon, 0)
        scores.append(score)
        cons.append([con])
    
    weights = tfco.find_best_candidate_distribution(np.array(scores), np.array(cons))

    return (clfs, weights)

def frank_wolfe_update(u_t, v_t, w_t, lambda_val, gamma, eta_t, n_class, p, X_train, y_train, vec_eta, epsilon, target):
    u_t_diff = np.copy(u_t)
    v_t_diff = np.copy(v_t)
    # for i in range(n_class):
        # u_t_diff[i, i] = u_t[i, i]/p[i]
        # v_t_diff[i, i] = v_t[i, i]/p[i]
        
    grad_u_t = grad_CM(u_t, v_t, w_t, lambda_val, n_class, p)
    grad_v_t = grad_FAIR(u_t, v_t, w_t, lambda_val, n_class, p)

    tilde_u, clf = LMO_CM(grad_u_t, X_train, y_train, vec_eta, n_class)
    tilde_v = LMO_FAIR(grad_v_t, p, epsilon, n_class, target)
    
    u_new = (1-gamma)*u_t + gamma*tilde_u
    v_new = (1-gamma)*v_t + gamma*tilde_v
    w_new = w_t + eta_t*(u_t_diff - v_t_diff)
    
    return (u_new, v_new, w_new, clf, tilde_u)
