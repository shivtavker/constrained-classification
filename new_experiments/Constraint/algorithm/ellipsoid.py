import numpy as np
from solvers.fwpost import frank_wolfe_post
from solvers.primal import solve_primal
from standard_funcs.helpers import compute_hmean, compute_cov_class, compute_cov_value
from standard_funcs.confusion_matrix import weight_confusion_matrix, get_confusion_vector
import tensorflow.compat.v1 as tf
import math
import tensorflow_constrained_optimization as tfco

def ellipsoid(X_train, y_train, vec_eta, T, n_class, epsilon, target, rho=0, inner_t=1000, eta_xi=5e-2):
    p = np.zeros((n_class,))
    for i in range(n_class):
        p[i] = (y_train == i).mean()
    n_lams = 2*n_class
    dim_n = 2*n_class + n_lams
    R_init = 1000
    Ps = [np.identity(dim_n)*R_init]
    x_centers = [np.ones(dim_n).reshape(dim_n, 1)]
    Cs = []
    clfs = []
    final_xis = []
    t = 0

    objectives = []
    constraints = []

    while t < T:
        P = Ps[-1]
        x_center = x_centers[-1]
        mu = x_center[:2*n_class].flatten()
        lams = x_center[2*n_class:].reshape(-1, 1)
        # if t > 0 and t%10 == 0:
        #     print("Iteration: ", t)
        #     print("Constraint Violation: ", constraints[-1])

        ## Check Constraints
        for i in range(n_lams):
            lam = lams[i]
            if(lam < 0):
                grad_vec = np.zeros(dim_n)
                grad_vec[2*n_class+i] = -1
                grad_vec = grad_vec.reshape(dim_n, 1)
                norm_const = np.sqrt(1/(np.dot(np.dot(grad_vec.T, P), grad_vec)))
                norm_grad = grad_vec*norm_const

                x_center_new = x_center - ((1/(dim_n+1))*np.dot(P, norm_grad))
                P_mod_part = (2/(dim_n+1))*np.matmul(np.dot(P, norm_grad), np.dot(norm_grad.T, P))
                P_new = ((dim_n**2)/(dim_n**2 - 1))*(P - P_mod_part)
                x_centers.append(x_center_new)
                Ps.append(P_new)
                break
        else:
            # if t > 0 and t%100 == 0:
            # #     # print(lams)
            # #     print("Mu Grad Norm Upper: ", np.linalg.norm(list(mu_grad)[:n_class]))
            # #     print("Mu Grad Norm Lower: ", np.linalg.norm(list(mu_grad)[n_class:]))
            #     print(t)
            #     print(constraints[-1][0])

            xi, clf, C = solve_primal(X_train, y_train, mu, inner_t, eta_xi, vec_eta, p, target, epsilon, lams, n_class)
            final_xis.append(xi)
            clfs.append(clf)
            Cs.append(C)
            objectives.append(compute_hmean(C))
            constraints.append([compute_cov_value(C, target) - epsilon])

            C_vec = get_confusion_vector(C)
            mu_grad = xi - C_vec 

            lams_grad = [0]*n_lams
            for i in range(n_class, 2*n_class):
                lams_grad[2*(i-n_class)] = xi[i] + (xi[i-n_class]*p[i-n_class]) - target[i-n_class] - epsilon
                lams_grad[2*(i-n_class)+1] = target[i-n_class] - (xi[i] + (xi[i-n_class]*p[i-n_class])) - epsilon
            
            grad_vec = -1*np.concatenate([mu_grad, lams_grad], axis=0).reshape((dim_n, 1)) 
            norm_const = np.sqrt((1/(np.dot(np.dot(grad_vec.T, P), grad_vec) + 1e-7)))

            if np.dot(np.dot(grad_vec.T, P), grad_vec) < 1e-7:
                print("Stopped at t = ", t)
                weights = tfco.find_best_candidate_distribution(
                    np.array(objectives), np.array(constraints)
                )

                return (clfs, weights)

            norm_grad = grad_vec*norm_const

            x_center_new = x_center - ((1/(dim_n+1))*np.dot(P, norm_grad))
            P_mod_part = (2/(dim_n+1))*np.matmul(np.dot(P, norm_grad), np.dot(norm_grad.T, P))
            P_new = ((dim_n**2)/(dim_n**2 - 1))*(P - P_mod_part)
            x_centers.append(x_center_new)
            Ps.append(P_new)
            t += 1
    
    # print("Done")

    weights = tfco.find_best_candidate_distribution(
        np.array(objectives), np.array(constraints)
    )

    return (clfs, weights)
