import numpy as np

class Loss_Grad_protected():
    def __init__(self, loss_name, unique_a, len_protected, y_train, lambda_val):
        self.M = np.concatenate((np.identity(4*unique_a), -1*np.identity(4*unique_a)), axis=1)
        self.unique_a = unique_a
        self.loss_name = loss_name
        self.len_protected = len_protected
        self.total_size = len(y_train)
        self.pi_0 = np.count_nonzero(y_train == 0)/len(y_train)
        self.pi_1 = 1 - self.pi_0
        self.lambda_val = lambda_val

    def grad_f(self, x_shape_a):
        x_shape_a_cm = x_shape_a[:self.unique_a]
        x_shape_a_fair = x_shape_a[self.unique_a:]

        weighted_sum_x_cm = np.average(x_shape_a_cm, axis=0, weights=self.len_protected/self.total_size)
        weighted_sum_x_fair = np.average(x_shape_a_fair, axis=0, weights=self.len_protected/self.total_size)
        
        sum_c00_cm = weighted_sum_x_cm[0][0]
        sum_c11_cm = weighted_sum_x_cm[1][1]
        sum_c00_fair = weighted_sum_x_fair[0][0]
        sum_c11_fair = weighted_sum_x_fair[1][1]
        grad_array_a = []

        for i in range(self.unique_a):
            x_cm = x_shape_a[i]
            x_fair = x_shape_a[self.unique_a+i]
            c_a00_cm = x_cm[0][0]
            c_a11_cm = x_cm[1][1]
            c_a00_fair = x_fair[0][0]
            c_a11_fair = x_fair[1][1]
            grad_c_a01_cm = 0
            grad_c_a10_cm = 0
            grad_c_a01_fair = 0
            grad_c_a10_fair = 0
            multiplicative_factor = self.len_protected[i]/self.total_size
            if(self.loss_name=="gmean"):
                grad_c_a00_cm = -1*sum_c11_cm*multiplicative_factor
                grad_c_a00_fair = -1*sum_c11_fair*multiplicative_factor
                grad_c_a11_cm = -1*sum_c00_cm*multiplicative_factor
                grad_c_a11_fair = -1*sum_c00_fair*multiplicative_factor
            elif(self.loss_name=="fmeasure"):
                grad_c_a00_cm = -2*sum_c11_cm*multiplicative_factor
                grad_c_a00_fair = -2*sum_c11_fair*multiplicative_factor
                grad_c_a11_cm = 2*(sum_c00_cm - 1)*multiplicative_factor
                grad_c_a11_fair = 2*(sum_c00_fair - 1)*multiplicative_factor
            elif(self.loss_name=="hmean"):
                grad_c_a00_cm = -1*(self.pi_0/(c_a00_cm**2+1e-7))*multiplicative_factor
                grad_c_a00_fair = -1*(self.pi_0/(c_a00_fair**2+1e-7))*multiplicative_factor
                grad_c_a11_cm = -1*(self.pi_1/(c_a11_cm**2+1e-7))*multiplicative_factor
                grad_c_a11_fair = -1*(self.pi_1/(c_a11_fair**2+1e-7))*multiplicative_factor
            elif(self.loss_name=="qmean"):
                grad_c_a00_cm = (1/self.pi_0)*((sum_c00_cm/self.pi_0) - 1)*multiplicative_factor
                grad_c_a00_fair = (1/self.pi_0)*((sum_c00_fair/self.pi_0) - 1)*multiplicative_factor
                grad_c_a11_cm = (1/self.pi_1)*((sum_c11_cm/self.pi_1) - 1)*multiplicative_factor
                grad_c_a11_fair = (1/self.pi_1)*((sum_c11_fair/self.pi_1) - 1)*multiplicative_factor
            elif(self.loss_name=="linear"):
                grad_c_a00_cm = -multiplicative_factor
                grad_c_a00_fair = -multiplicative_factor
                grad_c_a11_cm = -multiplicative_factor
                grad_c_a11_fair = -multiplicative_factor

            grad_array_a.append(
                [grad_c_a00_cm, grad_c_a01_cm, grad_c_a10_cm, grad_c_a11_cm,
                grad_c_a00_fair, grad_c_a01_fair, grad_c_a10_fair, grad_c_a11_fair]
            )
        grad_array_a = np.array(grad_array_a)
        return grad_array_a.flatten().T
    
    def phi_grad(self, x, y):
        x_vector = []
        for matrix in x:
            x_vector.append(matrix.reshape(4, 1))
        x_vector = np.array(x_vector)
        x_vector = x_vector.flatten().T
        
        grad_vector = self.grad_f(x) + np.dot(self.M.T, y).flatten() + self.lambda_val*np.dot(np.dot(self.M.T, self.M), x_vector).T
        
        return grad_vector.reshape(4*self.unique_a, 2)

class Loss_Grad_no_protected():
    def __init__(self, loss_name, y_train, lambda_val):
        self.M = np.concatenate((np.identity(4), -1*np.identity(4)), axis=1)
        self.loss_name = loss_name
        self.pi_0 = np.count_nonzero(y_train == 0)/len(y_train)
        self.pi_1 = 1 - self.pi_0
        self.lambda_val = lambda_val

    def grad_f(self, x):
        # print(x.shape)
        # print(x)
        c_00_cm = x[0][0]
        c_11_cm = x[1][1]
        # print(c_00_cm)
        c_00_fair = x[2][0]
        c_11_fair = x[3][1]

        grad_c_01_cm = 0
        grad_c_10_cm = 0
        grad_c_01_fair = 0
        grad_c_10_fair = 0
        if(self.loss_name=="gmean"):
            grad_c_00_cm = -1*c_11_cm
            grad_c_00_fair = -1*c_11_fair
            grad_c_11_cm = -1*c_00_cm
            grad_c_11_fair = -1*c_00_fair
        elif(self.loss_name=="fmeasure"):
            grad_c_00_cm = -2*c_11_cm
            grad_c_00_fair = -2*c_11_fair
            grad_c_11_cm = 2*(c_00_cm - 1)
            grad_c_11_fair = 2*(c_00_fair - 1)
        elif(self.loss_name=="hmean"):
            grad_c_00_cm = -1*(self.pi_0/(c_00_cm**2+1e-7))
            grad_c_00_fair = -1*(self.pi_0/(c_00_fair**2+1e-7))
            grad_c_11_cm = -1*(self.pi_1/(c_11_cm**2+1e-7))
            grad_c_11_fair = -1*(self.pi_1/(c_11_fair**2+1e-7))
        elif(self.loss_name=="qmean"):
            grad_c_00_cm = (1/self.pi_0)*((c_00_cm/self.pi_0) - 1)
            grad_c_00_fair = (1/self.pi_0)*((c_00_fair/self.pi_0) - 1)
            grad_c_11_cm = (1/self.pi_1)*((c_11_cm/self.pi_1) - 1)
            grad_c_11_fair = (1/self.pi_1)*((c_11_fair/self.pi_1) - 1)
        elif(self.loss_name=="linear"):
            grad_c_00_cm = -1
            grad_c_00_fair = -1
            grad_c_11_cm = -1
            grad_c_11_fair = -1

        grad_array = [grad_c_00_cm, grad_c_01_cm, grad_c_10_cm, grad_c_11_cm, grad_c_00_fair, grad_c_01_fair, grad_c_10_fair, grad_c_11_fair]
        return np.array(grad_array).T
    
    def phi_grad(self, x, y):
        x_vector = np.matrix.flatten(x)        
        grad_vector = self.grad_f(x) + np.dot(self.M.T, y).flatten() + self.lambda_val*np.dot(np.dot(self.M.T, self.M), x_vector.T).T
        
        return grad_vector.reshape(4, 2)