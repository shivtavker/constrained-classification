import numpy as np
from standard_funcs.loss_grad import Loss_Grad_protected, Loss_Grad_no_protected
from standard_funcs.helpers import get_unique_a, get_len_protected, get_protected_indices, compute_qmean_conf
from standard_funcs.confusion_matrix import get_confusion_matrix_threshold, get_confusion_matrix_from_loss_a
from solvers.lmo_cm import LMO_CM_protected, LMO_CM_no_protected
from solvers.lmo_fair import LMO_DP, LMO_EOdds, LMO_COV, LMO_KLD, LMO_EOpp
##for plot only
# import matplotlib.pyplot as plt
# from standard_funcs.randomized_classifiers import get_confusion_matrix_final_loss
# from standard_funcs.helpers import compute_gmean_conf, compute_EOpp_cms

class SBFW_protected():
    def __init__(self, X_train, y_train, vec_eta_1, loss_name, constraint_name, lambda_val, epsilon):
        self.constraint_name = constraint_name
        self.epsilon = epsilon
        self.x_train = X_train
        self.vec_eta_1 = vec_eta_1
        self.y_train = y_train 
        self.pi_0 = np.count_nonzero(y_train == 0)/len(y_train)
        self.pi_1 = 1 - self.pi_0
        self.unique_a = get_unique_a(X_train)
        self.ratio = []
        for a in range(self.unique_a):
            indices_a = X_train[:, 0] == a
            Y_label_a = y_train[indices_a]
            self.ratio.append(np.count_nonzero(Y_label_a==1)/len(Y_label_a))
        self.M = np.concatenate((np.identity(4*self.unique_a), -1*np.identity(4*self.unique_a)), axis=1)
        self.protected_indices = get_protected_indices(X_train)
        self.len_protected = get_len_protected(X_train)
        self.loss_Grad = Loss_Grad_protected(loss_name, self.unique_a, self.len_protected, y_train, lambda_val) 
    
    def split_frank_wolfe(self, x_t, y_t, gamma):
        grad_matrix = self.loss_Grad.phi_grad(x_t, y_t)
        matrix1, matrix2 = np.vsplit(grad_matrix, 2)
        grad_matrix1, grad_matrix2 = np.vsplit(matrix1, self.unique_a), np.vsplit(matrix2, self.unique_a)

        lmo_cm = LMO_CM_protected(grad_matrix1, self.x_train, self.y_train, self.unique_a, self.vec_eta_1)
        if(self.constraint_name == "DP"):
            s = np.concatenate((lmo_cm[0], LMO_DP(grad_matrix2, self.unique_a, self.epsilon)), axis = 0)
        elif(self.constraint_name == "EOdds"):
            s = np.concatenate((lmo_cm[0], LMO_EOdds(grad_matrix2, self.unique_a, self.epsilon)), axis = 0)
        elif(self.constraint_name == "EOpp"):
            s = np.concatenate((lmo_cm[0], LMO_EOpp(grad_matrix2, self.unique_a, self.epsilon, self.ratio)), axis = 0)
        else:
            raise SystemExit("Unknown Constraint")
            
        # search_result = line_search(functools.partial(phi, y=y_t), xk = x_t, pk = s - x_t)
        # gamma = search_result['alpha']
        # gamma = line_search(compute_gmean_cms_a, x_t, s)
        return ((1-gamma)*x_t + gamma*s, gamma, lmo_cm[1])

    def run_algo(self, eta_t_array, T):
        start_cm = []
        for i in range(self.unique_a):
            start_cm.append(
                get_confusion_matrix_threshold(0.2, self.x_train, self.y_train, self.vec_eta_1, "geq")
            )

        start_clf = []

        for i in range(self.unique_a):
            start_clf.append(
                np.matrix([[0, 1], [1, 0]])
            )

        start_cm = np.array(start_cm)

        start_fair = np.copy(start_cm)

        X = [np.concatenate((start_cm, start_fair), axis=0)]
        Y = [np.array([0.1, 0.1, 0.1, 0.1]*self.unique_a).reshape(-1, 1)]
        clfs = [start_clf]
        weights = np.zeros(T+1)
        weights[0] = 1
        train_cms = [get_confusion_matrix_from_loss_a(start_clf, self.x_train, self.y_train, self.unique_a, self.protected_indices, self.vec_eta_1)]
        train_loss = []
        train_constraint_violation = []

        for t in range(T):
            # if(t >= 0):
            #     final_cm_train = train_cms[-1]
            #     loss_train = compute_gmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(self.x_train)/len(self.y_train)), self.y_train)
            #     constraint_value_train = compute_EOpp_cms(final_cm_train, get_unique_a(self.x_train)) - self.epsilon
            #     train_loss.append(loss_train)
            #     train_constraint_violation.append(constraint_value_train)
            fw_res = self.split_frank_wolfe(X[-1], Y[-1], 2/(t+2))
            x_new = fw_res[0]
            gamma_new = fw_res[1]
            clf_new = fw_res[2]
            eta_t = eta_t_array[int(t*len(eta_t_array)/T)]
            x_new = x_new.reshape(8*self.unique_a, 1)
            y_new = Y[-1] + eta_t*np.dot(self.M, x_new)
            X.append(x_new.reshape(2*self.unique_a, 2, 2))
            Y.append(y_new)
            clfs.append(clf_new)
            weights = weights*(1-gamma_new) ## try to optimize here
            weights[t+1] = gamma_new
            ##plot only
            # train_cm_lmo = get_confusion_matrix_from_loss_a(clf_new, self.x_train, self.y_train, self.unique_a, self.protected_indices, self.vec_eta_1)
            # train_cms.append((1 - gamma_new)*train_cms[-1] + gamma_new*train_cm_lmo)

        answer, waste = np.vsplit(X[-1], 2)
        # np.save("sbfw-compas-3.npy", [train_loss, train_constraint_violation])
        # plt.plot(1 + np.arange(len(train_loss)), train_loss)
        # plt.xlabel("No Iterations")
        # plt.ylabel("Gmean Loss")
        # plt.title("Train Loss vs Iterations")
        # plt.show()
        # plt.plot(1 + np.arange(len(train_loss)), train_constraint_violation)
        # plt.xlabel("No Iterations")
        # plt.ylabel("Violation")
        # plt.title("Violation vs Iterations")
        # plt.show()

        return (clfs, weights)

        # print("Norm Difference: ", np.linalg.norm(answer-waste))
        # print("Gmean Loss: ", compute_gmean_conf(np.average(answer, axis=0, weights=len_protected/len(y_train))))
        # print("Equal Opp: ", compute_equal_opp_cms(answer))
        # print("===============================")

class SBFW_no_protected():
    def __init__(self, X_train, y_train, vec_eta_1, loss_name, constraint_name, lambda_val, epsilon):
        self.constraint_name = constraint_name
        self.epsilon = epsilon
        self.x_train = X_train
        self.vec_eta_1 = vec_eta_1
        self.y_train = y_train
        self.pi_0 = np.count_nonzero(y_train == 0)/len(y_train)
        self.pi_1 = 1 - self.pi_0
        self.M = np.concatenate((np.identity(4), -1*np.identity(4)), axis=1)
        self.loss_Grad = Loss_Grad_no_protected(loss_name, y_train, lambda_val) 
    
    def split_frank_wolfe(self, x_t, y_t, gamma):
        grad_matrix = self.loss_Grad.phi_grad(x_t, y_t)
        grad_matrix1, grad_matrix2 = np.vsplit(grad_matrix, 2)

        lmo_cm = LMO_CM_no_protected(grad_matrix1, self.x_train, self.y_train, self.vec_eta_1)
        if(self.constraint_name == "KLD"):
            s = np.concatenate((lmo_cm[0], LMO_KLD(grad_matrix2, self.epsilon, self.pi_0, self.pi_1)), axis = 0)
        elif(self.constraint_name == "COV"):
            s = np.concatenate((lmo_cm[0], LMO_COV(grad_matrix2, self.epsilon)), axis = 0)
        else:
            raise SystemExit("Unknown Constraint")
            
        # search_result = line_search(functools.partial(phi, y=y_t), xk = x_t, pk = s - x_t)
        # gamma = search_result['alpha']
        # gamma = line_search(compute_gmean_cms_a, x_t, s)
        return ((1-gamma)*x_t + gamma*s, gamma, lmo_cm[1])
    
    def run_algo(self, eta_t_array, T):
        start_clf = np.matrix([[0, 1], [1, 0]])
        start_cm = get_confusion_matrix_threshold(0.5, self.x_train, self.y_train, self.vec_eta_1)
        start_fair = np.copy(start_cm)

        X = [np.concatenate((start_cm, start_fair), axis=0)]
        Y = [np.array([0.1, 0.1, 0.1, 0.1]).reshape(-1, 1)]
        clfs = [start_clf]
        weights = np.zeros(T+1)
        weights[0] = 1

        for t in range(T):
            fw_res = self.split_frank_wolfe(X[-1], Y[-1], 2/(t+2))
            x_new = fw_res[0]
            gamma_new = fw_res[1]
            clf_new = fw_res[2]
            eta_t = eta_t_array[int(t*len(eta_t_array)/T)]
            x_new = x_new.reshape(8, 1)
            y_new = Y[-1] + eta_t*np.dot(self.M, x_new)
            X.append(x_new.reshape(4, 2))
            Y.append(y_new)
            clfs.append(clf_new)
            weights = weights*(1-gamma_new) ## try to optimize here
            weights[t+1] = gamma_new

        answer, waste = np.vsplit(X[-1], 2)

        return (clfs, weights)
