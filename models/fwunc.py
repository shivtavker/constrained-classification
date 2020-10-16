import numpy as np
from standard_funcs.helpers import *
from standard_funcs.confusion_matrix import get_confusion_matrix_threshold
from standard_funcs.loss_grad import Loss_Grad_no_protected
from solvers.lmo_cm import LMO_CM_threshold

class FWUnc_model():
    def __init__(self, X_train, y_train, vec_eta_1, loss_name):
        self.loss_name = loss_name
        self.x_train = X_train
        self.y_train = y_train
        self.vec_eta_1 = vec_eta_1
        self.loss_Grad = Loss_Grad_no_protected(loss_name, y_train, 0)

    def run_algorithm(self, T):
        start_cm = get_confusion_matrix_threshold(0.5, self.x_train, self.y_train, self.vec_eta_1, inequality="geq")
        X = [start_cm]
        thresholds = [0.5]

        for t in range(T):
            x = X[-1]
            threshold = thresholds[-1]
            combined_grad = np.concatenate((x, x), axis=0)
            # print(combined_grad)
            grad_vec = self.loss_Grad.grad_f(combined_grad)[:4]
            grad_matrix = grad_vec.reshape(2, 2).T
            lmo_cm = LMO_CM_threshold(grad_matrix, self.x_train, self.y_train, self.vec_eta_1)
            gamma = 2/(t+2)
            x_new = (1 - gamma)*x + gamma*lmo_cm[0]
            threshold_new = (1 - gamma)*threshold + gamma*lmo_cm[1]
            X.append(x_new)
            thresholds.append(threshold_new)
        return thresholds[-1]