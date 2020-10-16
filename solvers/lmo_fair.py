import numpy as np
from standard_funcs.helpers import get_row
from scipy.optimize import linprog

### Linear Minimization Oracles for Constraints
### Constraint <= epsilon
### unique_a -> number of unique values in Protexted attribute
def LMO_DP(grad_matrix, unique_a, epsilon):
    """
    grad_matrix : shape n x n x unique_a
    unique_a: int
    epsilon: int
    """

    ### Flatten the Matrix
    grad_array = []
    for i in range(unique_a):
        grad_a = np.matrix.flatten(np.array(grad_matrix[i]))
        grad_array.append(grad_a)

    ### Linear Programming Solver
    c = np.matrix.flatten(np.array(grad_array))
    common_sub = (1/unique_a)*get_row(np.arange(unique_a), [1, 3], unique_a)
    A_ub = []
    for i in range(unique_a):
        row_a1 = get_row([i], [1, 3], unique_a) - common_sub
        row_a2 = -1*row_a1
        A_ub.append(row_a1)
        A_ub.append(row_a2)
    b = np.array([epsilon]*len(A_ub))
    A_ub = np.array(A_ub)
    soln = linprog(c=c, A_ub = A_ub, b_ub = b, bounds=(0, 1), method='interior-point', options={'tol':1e-5, 'maxiter': 100})
    soln_full_vector = soln['x']
    
    ### Convert soln to required matrix form
    soln_shape_4 = np.split(soln_full_vector, unique_a)
    answer = []
    for i in range(unique_a):
        answer.append(soln_shape_4[i].reshape((2, 2)))
    return answer

def LMO_EOdds(grad_matrix, unique_a, epsilon):
        # Ax <= b
    grad_array = []
    
    for i in range(unique_a):
        grad_a = np.matrix.flatten(np.array(grad_matrix[i]))
        grad_array.append(grad_a)

    c = np.matrix.flatten(np.array(grad_array))
    common_sub_0 = (1/unique_a)*get_row(np.arange(unique_a), [1], unique_a)
    common_sub_1 = (1/unique_a)*get_row(np.arange(unique_a), [3], unique_a)
    
    A_ub = []
    
    for i in range(unique_a):
        row_a1 = get_row([i], [1], unique_a) - common_sub_0
        row_a2 = -1*row_a1
        row_a3 = get_row([i], [3], unique_a) - common_sub_1
        row_a4 = -1*row_a3
        A_ub.append(row_a1)
        A_ub.append(row_a2)
        A_ub.append(row_a3)
        A_ub.append(row_a4)
        
    b = np.array([epsilon]*len(A_ub))
    
    A_ub = np.array(A_ub)
    
    soln = linprog(c=c, A_ub = A_ub, b_ub = b, bounds=(0, 1), method='interior-point', options={'tol':1e-5, 'maxiter': 100})
    # soln_24
    soln_full_vector = soln['x']
    soln_shape_4 = np.split(soln_full_vector, unique_a)
    answer = []
    for i in range(unique_a):
        answer.append(soln_shape_4[i].reshape((2, 2)))
    return answer

def LMO_EOpp(grad_matrix, unique_a, epsilon, ratio):
    grad_array = []
    for i in range(unique_a):
        grad_a = np.matrix.flatten(np.array(grad_matrix[i]))
        grad_array.append(grad_a)

    c = np.matrix.flatten(np.array(grad_array))
    common_sub_1 = (1/unique_a)*get_row(np.arange(unique_a), [3], unique_a)
    for i in range(unique_a):
        common_sub_1[(4*i)+3] = common_sub_1[(4*i)+3]/ratio[i]
    A_ub = []
    
    for i in range(unique_a):
        row_a3 = (get_row([i], [3], unique_a)/ratio[i] - common_sub_1)
        row_a4 = -1*row_a3
        A_ub.append(row_a3)
        A_ub.append(row_a4)
        
    # b = np.array([epsilon*ratio[i], epsilon*ratio[i] for i in range(unique_a)])
    b = []
    for i in range(unique_a):
        b.append(epsilon)
        b.append(epsilon)
    b = np.array(b)
    
    A_ub = np.array(A_ub)
    
    soln = linprog(c=c, A_ub = A_ub, b_ub = b, bounds=(0, 1), method='interior-point', options={'tol':1e-5, 'maxiter': 100})
    # soln_24
    soln_full_vector = soln['x']
    soln_shape_4 = np.split(soln_full_vector, unique_a)
    answer = []
    for i in range(unique_a):
        answer.append(soln_shape_4[i].reshape((2, 2)))
    return answer

def LMO_KLD(grad_matrix, epsilon, pi_0, pi_1):
    grad_vec = -1*np.ravel(grad_matrix.flatten())
    w0 = min(grad_vec[0], grad_vec[2])
    w1 = min(grad_vec[1], grad_vec[3])
    if(w0 < 0 and w1 < 0):
        lambda_opt = np.exp(pi_0*np.log(abs(w0) + 1e-7) + pi_1*np.log(abs(w1) + 1e-7) - epsilon)
        soln_4 = np.zeros(4)
        if(grad_vec[0] < grad_vec[2]):
            soln_4[2] = abs(lambda_opt*pi_0/w0)
            soln_4[0] = 0
        else:
            soln_4[0] = abs(lambda_opt*pi_0/w0)
            soln_4[2] = 0
        if(grad_vec[1] < grad_vec[3]):
            soln_4[3] = abs(lambda_opt*pi_1/w1)
            soln_4[1] = 0
        else:
            soln_4[1] = abs(lambda_opt*pi_1/w1)
            soln_4[3] = 0
    else:
        corner_1_x = np.exp((pi_0*np.log(pi_0) + pi_1*np.log(pi_1) - epsilon)/pi_0)
        corner_1_y = 1
        corner_2_x = 1
        corner_2_y = np.exp((pi_0*np.log(pi_0) + pi_1*np.log(pi_1) - epsilon)/pi_1)
        corner_points = [(corner_1_x, corner_1_y), (corner_2_x, corner_2_y), (1, 1)]
        if(w0 >= 0 and w1 >= 0):
            corner_point = corner_points[2]
        elif(w0 >= 0 and w1 < 0):
            corner_point = corner_points[1]
        else:
            corner_point = corner_points[0]
            
        soln_4 = np.zeros(4)
        if(grad_vec[0] < grad_vec[2]):
            soln_4[2] = corner_point[0]
            soln_4[0] = 0
        else:
            soln_4[0] = corner_point[0]
            soln_4[2] = 0
        if(grad_vec[1] < grad_vec[3]):
            soln_4[3] = corner_point[1]
            soln_4[1] = 0
        else:
            soln_4[1] = corner_point[1]
            soln_4[3] = 0
    
    answer = soln_4.reshape(2, 2)
    return answer

def LMO_COV(grad_matrix, epsilon):
    grad_vec = grad_matrix.reshape(1, 4)
    A_ub = [np.array([0, 1, 0, 1])]
    A_ub = np.array(A_ub)
    b = [epsilon]

    soln = linprog(c=grad_vec, A_ub = A_ub, b_ub = b, bounds=(1e-5, 1), method='interior-point', options={'tol':1e-5, 'maxiter': 100})
    soln_4 = soln['x']
    answer = soln_4.reshape((2, 2))
    return answer
