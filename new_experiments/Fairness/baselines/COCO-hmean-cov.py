import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from complex_performance_metrics.models.constrained import COCOClassifier

params_data = [
    ("data/adult_data.npy", "Adult", "hmean", "cov", 0.25, 20),
    ("data/compas_data.npy", "COMPAS", "hmean", "cov", 0.25, 20),
    ("data/crimes_data.npy", "Crimes", "hmean", "cov", 0.25, 20),
    ("data/default_data.npy", "Default", "hmean", "cov", 0.25, 20),
    ("data/lawschool_data.npy", "Lawschool", "hmean", "cov", 0.25, 20)
]

def compute_qmean_conf(confusion_matrix, pi_0, pi_1):
    return np.sqrt(0.5*((1 - confusion_matrix[0][0]/pi_0)**2 + (1 - confusion_matrix[1][1]/pi_1)**2))

for param_data in params_data:
    data_dict = np.load(param_data[0]).item()
    eta = param_data[5]
    X_train = data_dict.get('X_train')
    y_train = data_dict.get('y_train')
    X_test = data_dict.get('X_test')
    y_test = data_dict.get('y_test')

    n_class = len(np.unique(y_train))
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    pi_0_train = np.count_nonzero(y_train == 0)/len(y_train)
    pi_1_train = 1 - pi_0_train
    pi_0_test = np.count_nonzero(y_test == 0)/len(y_test)
    pi_1_test = 1 - pi_0_test

    ## Logistic Regression for Estimate
    params_tune = [(0.01, "l1"), (1, "l1"), (10, "l1"), (0.01, "l2"), (1, "l2"), (10, "l2")]
    qmean_tries = np.zeros(len(params_tune))

    # def compute_qmean(Y_label, Y_pred):
    #     pi_array = []
    #     for i in range(n_class):
    #         pi_a = np.count_nonzero(Y_label == i)
    #         pi_array.append(pi_a)
        
    #     confusion_matrix = np.zeros(shape=(n_class, n_class))
    #     np.add.at(confusion_matrix, (Y_label.astype(int), Y_pred.astype(int)), 1)
    #     sum_sq_form = 0
    #     for i in range(n_class):
    #         sum_sq_form += (1 - (confusion_matrix[i][i]/pi_array[i]))**2
    #     return np.sqrt(sum_sq_form/n_class)

    for i in range(len(params_tune)):
        C_param = params_tune[i][0]
        penalty_param = params_tune[i][1]
        lr = LogisticRegression(solver="liblinear", 
                                max_iter=1000, 
                                C=C_param, 
                                penalty=penalty_param
                                ).fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        qmean_tries[i] = accuracy_score(y_train, y_pred)
        
    best_param = params_tune[np.argmax(qmean_tries)]

    lr = LogisticRegression(solver="liblinear", 
                                max_iter=1000, 
                                C=best_param[0], 
                                penalty=best_param[1]
                                ).fit(X_train, y_train)

    epsilon = param_data[4]

    best_eta = eta
    coco = COCOClassifier(param_data[2], param_data[3])
    coco.fit(X_train, y_train, epsilon, best_eta, num_outer_iter=400, num_inner_iter=20, cpe_model=lr)

    # print("Dataset: " + str(param_data[1]) + " | Loss: " + str(param_data[2]) + " | Epsilon = " + str(epsilon) + " | eta = " + str(best_eta))
    # conf_matrix_test = coco.evaluate_conf(X_test, y_test)
    # conf_matrix_train = coco.evaluate_conf(X_train, y_train)
    # print("Train Qmean Loss: ", compute_qmean_conf(conf_matrix_train[0], pi_0_train, pi_1_train))
    # print("Train Hmean Loss: ", coco.evaluate_loss(X_train, y_train, X_train[:, 0]))
    # print("Train Constraint: ", coco.evaluate_cons(X_train, y_train))
    print("%.3f(%.3f)"% (coco.evaluate_loss(X_train, y_train, X_train[:, 0]), coco.evaluate_cons(X_train, y_train)))
    # print("Test Qmean Loss: ", compute_qmean_conf(conf_matrix_test[0], pi_0_test, pi_1_test))
    # print("Test Hmean Loss: ", coco.evaluate_loss(X_test, y_test, X_test[:, 0]))
    # print("Test Constraint: ", coco.evaluate_cons(X_test, y_test))   
    print("%.3f(%.3f)"% (coco.evaluate_loss(X_test, y_test, X_test[:, 0]), coco.evaluate_cons(X_test, y_test))) 

    # conf_matrix = coco.evaluate_conf(X_test, y_test, z_ts=X_test[:, 0])

    # # print(conf_matrix)

    # print("Q mean Loss: ", compute_qmean_conf(conf_matrix[0], pi_0_test, pi_1_test))
    # print("Train Constraint: ", coco.evaluate_cons(X_train, y_train, z_ts=X_train[:, 0]))
    # print("Test Constraint: ", coco.evaluate_cons(X_test, y_test, z_ts=X_test[:, 0]))

