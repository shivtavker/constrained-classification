import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.threeplayer import threeplayer
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_hmean, compute_cov_value
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco

### Data Loading
for d in ["covtype"]:
    t_train_scores = []
    t_train_cons = []
    t_test_scores = []
    t_test_cons = []
    
    eps = 0.01

    for T in [1000]:
        data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
        X_data = data_dict['X']
        Y_data = data_dict['Y']

        train_scores = []
        train_cons = []
        test_scores = []
        test_cons = []

        for global_i in range(3):
            X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=global_i)

            ### Number of Classes
            n_class = len(np.unique(y_train))

            ### Training CPE Model
            vec_eta = get_vec_eta(X_train, y_train)
            
            ### Setting Target
            target = np.zeros((n_class,))
            for i in range(n_class):
                target[i] = (y_train == i).mean()

            lr_range = [0.005, 0.05, 0.1]
            grid = [(xx, yy) for xx in lr_range for yy in lr_range]
            objectives = []
            violations = []

            for eta_param, eta_xi in grid:
                ### Getting the Classifiers and Weights
                clfs, weights = threeplayer(X_train, y_train, vec_eta, T, n_class, eps, target, eta_lam=eta_param, eta_mu=eta_param, eta_xi=eta_xi)

                ### Evaluate Performance on Train Data
                train_conf = weight_confusion_matrix(X_train, y_train, clfs, weights, n_class, vec_eta)
                train_score = 1 - compute_hmean(train_conf)

                p = np.zeros((n_class,))
                for i in range(n_class):
                    p[i] = (y_train == i).mean()

                train_con = compute_cov_value(train_conf, p)

                objectives.append(1-train_score)
                violations.append([train_con-eps])
            
            best_index = tfco.find_best_candidate_index(np.array(objectives), np.array(violations), rank_objectives = False)

            best_eta_param, best_eta_xi = grid[best_index]

            # print(objectives)
            # print(violations)

            clfs, weights = threeplayer(X_train, y_train, vec_eta, T, n_class, eps, target, eta_lam=best_eta_param, eta_mu=best_eta_param, eta_xi=best_eta_xi)

            ### Evaluate Performance on Train Data
            train_conf = weight_confusion_matrix(X_train, y_train, clfs, weights, n_class, vec_eta)
            train_score = 1 - compute_hmean(train_conf)

            p = np.zeros((n_class,))
            for i in range(n_class):
                p[i] = (y_train == i).mean()

            train_con = compute_cov_value(train_conf, p)

            ### Evaluate Performance on Test Data
            test_conf = weight_confusion_matrix(X_test, y_test, clfs, weights, n_class, vec_eta)
            test_score = 1 - compute_hmean(test_conf)

            p = np.zeros((n_class,))
            for i in range(n_class):
                p[i] = (y_test == i).mean()

            test_con = compute_cov_value(test_conf, p)

            train_scores.append(train_score)
            train_cons.append(train_con)
            test_scores.append(test_score)
            test_cons.append(test_con)
            # print(global_i)
        print(train_scores)
        print(train_cons)
        print(test_scores)
        print(test_cons)

        t_train_scores.append(train_scores)
        t_train_cons.append(train_cons)
        t_test_scores.append(test_scores)
        t_test_cons.append(test_cons)

    np.save("./Results/3pplug_" + d + "_train_scores.npy", np.array(t_train_scores[0]))
    np.save("./Results/3pplug_" + d + "_test_scores.npy", np.array(t_test_scores[0]))

    np.save("./Results/3pplug_" + d + "_train_cons.npy", np.array(t_train_cons[0]))
    np.save("./Results/3pplug_" + d + "_test_cons.npy", np.array(t_test_cons[0]))
