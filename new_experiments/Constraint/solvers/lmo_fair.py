import numpy as np
from standard_funcs.helpers import get_row
from scipy.optimize import linprog

### Linear Minimization Oracles for Constraints
### Constraint <= epsilon

def LMO_FAIR(grad_vector, p, epsilon, n_class, target):
    A_ub = []
    b = []
    
    for i in range(n_class):
        vec = np.zeros(2*n_class)
        vec[i] = 1
        vec[i+n_class] = 1
        A_ub.append(vec)
        b.append(target[i] + epsilon)
        vec = np.zeros(2*n_class)
        vec[i] = -1
        vec[i+n_class] = -1
        A_ub.append(vec)
        b.append(epsilon - target[i])

    soln = linprog(c=grad_vector, A_ub = A_ub, b_ub = b, bounds=(0, 1))
    soln_vec = soln['x']
    
    return soln_vec

    # Ax <= b
    # for i in range(n_class):
    #     grad_matrix[i, i] = grad_matrix[i, i]*p[i]
    # grad_vec = grad_matrix.reshape(1, n_class*n_class)
    # A_ub = [] #[np.array([0, 1, 0, 1])]
    # b = []
    # for k in range(n_class):
    #     a_ub = np.zeros(n_class*n_class)
    #     neg_a_ub = np.zeros(n_class*n_class)
    #     for i in range(n_class):
    #         a_ub[i*n_class + k] = 1
    #         neg_a_ub[i*n_class + k] = -1
    #         if(i == k):
    #             # a_ub[i*n_class + k] = 1*p[k]
    #             # neg_a_ub[i*n_class + k] = -1*p[k]
    #             a_ub[i*n_class + k] = 1
    #             neg_a_ub[i*n_class + k] = -1
        
    #     A_ub.append(a_ub)
    #     b.append(target[k] + epsilon)
    #     A_ub.append(neg_a_ub)
    #     b.append(epsilon - target[k])

    # # soln = linprog(c=grad_vec, A_ub = A_ub, b_ub = b, bounds=(1e-5, 1/(min(p)+1e-5)))
    # soln = linprog(c=grad_vec, A_ub = A_ub, b_ub = b, bounds=(0, 1))

    # soln_vec = soln['x']
    # answer = soln_vec.reshape((n_class, n_class))
    # # for i in range(n_class):
    # #     answer[i, i] *= p[i]
    # return answer
