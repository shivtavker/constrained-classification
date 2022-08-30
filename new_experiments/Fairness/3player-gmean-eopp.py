import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.stats as stats
import math

import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco

for name in ["lawschool"]:
    data_dict = np.load("data/" + name +"_data.npy", allow_pickle=True).item()
    X_train = data_dict.get('X_train')
    y_train = data_dict.get('y_train')
    X_test = data_dict.get('X_test')
    y_test = data_dict.get('y_test')
    n_class = 2

    X = np.vstack((X_train, X_test))
    Y = np.hstack((y_train, y_test))

    train_scores = []
    test_scores = []
    train_cons_val = []
    test_cons_val = []

    for global_i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=global_i, test_size=0.3)
        # dict_data = {
        #     'X_train': X_train,
        #     'X_test': X_test,
        #     'y_train': y_train,
        #     'y_test': y_test
        # }
        # np.save("adult-x-train.npy", X_train, allow_pickle=True)
        # np.save("adult-x-test.npy", X_test, allow_pickle=True)
        # np.save("adult-y-train.npy", y_train, allow_pickle=True)
        # np.save("adult-y-test.npy", y_test, allow_pickle=True)


        def get_confusion_matrix(Y_label, Y_pred, n_class):
            confusion_matrix = np.zeros(shape=(n_class, n_class))
            np.add.at(confusion_matrix, (Y_label.astype(int), Y_pred.astype(int)), 1)
            return confusion_matrix/len(Y_label)

        def get_confusion_matrix_protected(Y_label, Y_pred, ind):
            Y_label_a = Y_label[ind]
            Y_pred_a = Y_pred[ind]
            return get_confusion_matrix(Y_label_a, Y_pred_a)

        def compute_gmean(C):
            """
            Attributes:
                C (array-like, dtype=float, shape=(n,n)): Confusion matrix

            Returns:
                loss (float): H-mean loss
            """
            n = C.shape[0]
            gm_part = 1
            for i in range(n):
                gm_part *= C[i, i]/C[i, :].sum()
            return 1 - (gm_part)**(1/n)

        def compute_DP(Cs, nus):
            individual_scores = []
            for i in range(2):
                individual_scores.append(1/nus[i] * (np.sum(Cs[i][:, 1])))
            
            individual_scores = np.array(individual_scores)
            individual_scores = np.abs(individual_scores - np.mean(individual_scores))
            
            return max(individual_scores)

        def compute_EOpp(Cs, pia):
            individual_scores = []
            for i in range(2):
                individual_scores.append(1/pia[i] * (Cs[i][1, 1]))
            
            individual_scores = np.array(individual_scores)
            individual_scores = np.abs(individual_scores - np.mean(individual_scores))
            
            return max(individual_scores)

        def compute_EOpp00(xi, pia):
            xi_A, xi_B = np.vsplit(xi, 2)
            
            pred_a1 = xi_A[3][0]
            pred_b1 = xi_B[3][0]
            
            return pred_a1/(2*pia[0]) - pred_b1/(2*pia[1])

        def compute_EOpp01(xi, pia):
            xi_A, xi_B = np.vsplit(xi, 2)
            
            pred_a1 = xi_A[3][0]
            pred_b1 = xi_B[3][0]
            
            return pred_b1/(2*pia[1]) - pred_a1/(2*pia[0])

        def compute_EOpp10(xi, pia):
            xi_A, xi_B = np.vsplit(xi, 2)
            
            pred_a1 = xi_A[3][0]
            pred_b1 = xi_B[3][0]
            
            return pred_b1/(2*pia[1]) - pred_a1/(2*pia[0])

        def compute_EOpp11(xi, pia):
            xi_A, xi_B = np.vsplit(xi, 2)
            
            pred_a1 = xi_A[3][0]
            pred_b1 = xi_B[3][0]
            
            return pred_a1/(2*pia[0]) - pred_b1/(2*pia[1])

        def get_vec_eta(X_train, y_train):
            lr = LogisticRegressionCV(
                solver="newton-cg",
                max_iter=100000,
                cv=2,
                tol=1e-3,
                multi_class="auto"
                    ).fit(X_train, y_train)

            def vec_eta(X):
                return lr.predict_proba(X)

            return vec_eta

        def get_confusion_matrix_from_loss(L_t, X_train, y_train, vec_eta, n_class):
            confusion_matrix = np.zeros(shape=(n_class, n_class))
            Y_pred = np.zeros(len(y_train))
            Pr_matrix = vec_eta(X_train).T
            ### Loss if predicted class = row i, data_point j
            Loss_matrix = []
            
            for i in range(n_class):
                L_t_vec = L_t[:, i].T
                Loss_i = np.dot(L_t_vec, Pr_matrix)
                Loss_matrix.append(Loss_i)
            Loss_matrix = np.array(Loss_matrix)
            
            Y_pred = np.argmin(Loss_matrix, axis=0)
            np.add.at(confusion_matrix, (y_train.astype(int), Y_pred.astype(int)), 1)
            return confusion_matrix/len(X_train)

        p = np.zeros((n_class,))
        for i in range(n_class):
            p[i] = (y_train == i).mean()

        train_ind_a = []
        train_nu_a = []
        test_ind_a = []
        test_nu_a = []

        for i in range(2):
            train_ind_i = X_train[:, 0] == i
            train_nu_i = np.count_nonzero(train_ind_i)/len(X_train)
            test_ind_i = X_test[:, 0] == i
            test_nu_i = np.count_nonzero(test_ind_i)/len(X_test)
            
            train_ind_a.append(train_ind_i)
            train_nu_a.append(train_nu_i)
            
            test_ind_a.append(test_ind_i)
            test_nu_a.append(test_nu_i)

        train_pia_a = []
        test_pia_a = []

        train_ind_pos = y_train == 1
        test_ind_pos = y_test == 1

        for i in range(2):
            train_pia_i = np.count_nonzero(np.logical_and(train_ind_a[i], train_ind_pos))/np.count_nonzero(train_ind_a[i])
            train_pia_a.append(train_pia_i)
            
            test_pia_i = np.count_nonzero(np.logical_and(test_ind_a[i], test_ind_pos))/np.count_nonzero(test_ind_a[i])
            test_pia_a.append(test_pia_i)

        vec_eta = get_vec_eta(X_train, y_train)
        n_class = len(np.unique(y_train))

        nus = train_nu_a
        pia = train_pia_a

        L = 1 - np.eye(n_class)
        print(np.trace(get_confusion_matrix_from_loss(L, X_train, y_train, vec_eta, n_class)))

        ## Flatten all matrices
        d = n_class**2
        n_lams = 4

        def grad_gmean(C, n_class):
            W = np.zeros((n_class, n_class))
            
            W[0, 0] = -1*C[1, 1]
            W[1, 1] = -1*C[0, 0]
        
            return W

        def grad_gmean_xi(xi, nus):
            xi_A, xi_B = np.vsplit(xi, 2)
            cm = (nus[0]*xi_A + nus[1]*xi_B).reshape(2, 2)
            grad_matrix = grad_gmean(cm, n_class)
            grad_vec = grad_matrix.reshape(d, 1)
            
            grad_xi_A = nus[0]*grad_vec
            grad_xi_B = nus[1]*grad_vec
            
            return np.vstack((grad_xi_A, grad_xi_B))

        def grad_EOpp00(xi, pia):
            grad_vec = np.zeros(2*d)
            grad_vec[3] = p[1]*1/(2*pia[0])
            grad_vec[d+3] = -p[1]/(2*pia[1])
            
            return grad_vec.reshape(2*d, 1)

        def grad_EOpp01(xi, pia):
            grad_vec = np.zeros(2*d)
            grad_vec[3] = -p[1]/(2*pia[0])
            grad_vec[d+3] = p[1]/(2*pia[1])
            
            return grad_vec.reshape(2*d, 1)

        def grad_EOpp10(xi, pia):
            grad_vec = np.zeros(2*d)
            grad_vec[3] = -p[1]/(2*pia[0])
            grad_vec[d+3] = p[1]/(2*pia[1])
            
            return grad_vec.reshape(2*d, 1)

        def grad_EOpp11(xi, pia):
            grad_vec = np.zeros(2*d)
            grad_vec[3] = p[1]/(2*pia[0])
            grad_vec[d+3] = -p[1]/(2*pia[1])
            
            return grad_vec.reshape(2*d, 1)

        def project_A(matrix):
            return matrix.clip(0, 1)

        epsilon = 0.05

        for T in [5000]:
            losses = []
            constraint_violations = []
            params_tried = [(1e-3, 1e-3), (0.01, 1e-2), (0.5, 1e-2), (0.1, 0.1), (0.5, 0.5)]
            
            for j in range(len(params_tried)):
                eta_param, eta = params_tried[j]

                mus = [5*np.ones(shape = (2*n_class*2, 1))]
                clfs = []
                Cs = []
                xis = [0.5*np.ones(shape = (2*n_class*2, 1))]
                lams = [1]*n_lams
                objectives = []
                constraints = []

                for t in range(T):
                    mu = mus[-1]
                    mu_A, mu_B = np.vsplit(mu, 2)
                    L_A, L_B = -mu_A.reshape(2, 2), -mu_B.reshape(2, 2)

                    for i in range(n_class):
                        L_A[i, i] = L_A[i, i]/p[i]

                    for i in range(n_class):
                        L_B[i, i] = L_B[i, i]/p[i]

                    C_A = get_confusion_matrix_from_loss(L_A, X_train[train_ind_a[0]], y_train[train_ind_a[0]], vec_eta, n_class)
                    C_B = get_confusion_matrix_from_loss(L_B, X_train[train_ind_a[1]], y_train[train_ind_a[1]], vec_eta, n_class)
                    C = np.vstack((C_A.reshape(d, 1), C_B.reshape(d, 1)))

                    C_A_normalized = np.copy(C_A)
                    C_B_normalized = np.copy(C_B)

                    for i in range(n_class):
                        C_A_normalized[i, i] = C_A[i, i]/p[i]
                    for i in range(n_class):
                        C_B_normalized[i, i] = C_B[i, i]/p[i]

                    C_normalized = np.vstack((C_A_normalized.reshape(d, 1), C_B_normalized.reshape(d, 1)))

                    clfs.append((L_A, L_B))
                    Cs.append(C)

                    cm = (nus[0]*C_A + nus[1]*C_B).reshape(2, 2)
                    objectives.append(compute_gmean(cm))
                    constraints.append([max(compute_EOpp([C_A.reshape(2, 2), C_B.reshape(2, 2)], pia) - epsilon, 0)])

                    xi = xis[-1]

                    xi_unscaled = np.copy(xi)
                    xi_unscaled[0][0] = xi[0][0]*p[0]
                    xi_unscaled[3][0] = xi[3][0]*p[1]
                    xi_unscaled[4][0] = xi[4][0]*p[0]
                    xi_unscaled[7][0] = xi[7][0]*p[1]

                    grad_xi = grad_gmean_xi(xi, nus) + mu

                    grad_xi += lams[0]*grad_EOpp00(xi, pia) + lams[1]*grad_EOpp01(xi, pia) + lams[2]*grad_EOpp10(xi, pia) + lams[3]*grad_EOpp11(xi, pia)

                    xi_new = project_A(xi - eta*(grad_xi))
                    mu_new = mu + eta_param*(xi_new - C_normalized)

                    lams_grad = [0]*n_lams
                    lams_grad[0] = compute_EOpp00(xi_unscaled, pia) - epsilon
                    lams_grad[1] = compute_EOpp01(xi_unscaled, pia) - epsilon
                    lams_grad[2] = compute_EOpp10(xi_unscaled, pia) - epsilon
                    lams_grad[3] = compute_EOpp11(xi_unscaled, pia) - epsilon

                    for i in range(n_lams):
                        lams[i] = max(lams[i] + eta_param*(lams_grad[i]), 0)

                    mus.append(mu_new)
                    xis.append(xi_new)

                weights = tfco.find_best_candidate_distribution(
                    np.array(objectives), np.array(constraints)
                )

                net_xi = np.zeros(shape=Cs[0].shape)

                for i in range(len(Cs)):
                    xi = Cs[i]
                    net_xi += weights[i]*xi

                xi_A, xi_B = np.vsplit(net_xi, 2)
                net_cm = (nus[0]*xi_A + nus[1]*xi_B).reshape(2, 2)

                loss = compute_gmean(net_cm)
                violation = compute_EOpp([xi_A.reshape(2, 2), xi_B.reshape(2, 2)], pia) - epsilon

                losses.append(loss)
                constraint_violations.append([violation])
                
            best_index = tfco.find_best_candidate_index(np.array(losses), np.array(constraint_violations), rank_objectives=False)

        for T in [8000]:    
            eta_param, eta = params_tried[best_index]

            mus = [5*np.ones(shape = (2*n_class*2, 1))]
            clfs = []
            Cs = []
            xis = [0.5*np.ones(shape = (2*n_class*2, 1))]
            lams = [1]*n_lams
            objectives = []
            constraints = []

            for t in range(T):
                mu = mus[-1]
                mu_A, mu_B = np.vsplit(mu, 2)
                L_A, L_B = -mu_A.reshape(2, 2), -mu_B.reshape(2, 2)
                
                for i in range(n_class):
                    L_A[i, i] = L_A[i, i]/p[i]
                    
                for i in range(n_class):
                    L_B[i, i] = L_B[i, i]/p[i]
                
                C_A = get_confusion_matrix_from_loss(L_A, X_train[train_ind_a[0]], y_train[train_ind_a[0]], vec_eta, n_class)
                C_B = get_confusion_matrix_from_loss(L_B, X_train[train_ind_a[1]], y_train[train_ind_a[1]], vec_eta, n_class)
                C = np.vstack((C_A.reshape(d, 1), C_B.reshape(d, 1)))

                C_A_normalized = np.copy(C_A)
                C_B_normalized = np.copy(C_B)

                for i in range(n_class):
                    C_A_normalized[i, i] = C_A[i, i]/p[i]
                for i in range(n_class):
                    C_B_normalized[i, i] = C_B[i, i]/p[i]

                C_normalized = np.vstack((C_A_normalized.reshape(d, 1), C_B_normalized.reshape(d, 1)))

                clfs.append((L_A, L_B))
                Cs.append(C)

                cm = (nus[0]*C_A + nus[1]*C_B).reshape(2, 2)
                objectives.append(compute_gmean(cm))
                constraints.append([compute_EOpp([C_A.reshape(2, 2), C_B.reshape(2, 2)], pia) - epsilon])

                xi = xis[-1]

                xi_unscaled = np.copy(xi)
                xi_unscaled[0][0] = xi[0][0]*p[0]
                xi_unscaled[3][0] = xi[3][0]*p[1]
                xi_unscaled[4][0] = xi[4][0]*p[0]
                xi_unscaled[7][0] = xi[7][0]*p[1]

                grad_xi = grad_gmean_xi(xi, nus) + mu

                grad_xi += lams[0]*grad_EOpp00(xi, pia) + lams[1]*grad_EOpp01(xi, pia) + lams[2]*grad_EOpp10(xi, pia) + lams[3]*grad_EOpp11(xi, pia)

                xi_new = project_A(xi - eta*(grad_xi))
                mu_new = mu + eta_param*(xi_new - C_normalized)

                lams_grad = [0]*n_lams
                lams_grad[0] = compute_EOpp00(xi_unscaled, pia) - epsilon
                lams_grad[1] = compute_EOpp01(xi_unscaled, pia) - epsilon
                lams_grad[2] = compute_EOpp10(xi_unscaled, pia) - epsilon
                lams_grad[3] = compute_EOpp11(xi_unscaled, pia) - epsilon

                for i in range(n_lams):
                    lams[i] = max(lams[i] + eta_param*(lams_grad[i]), 0)

                mus.append(mu_new)
                xis.append(xi_new)
            # print(lams)

            weights = tfco.find_best_candidate_distribution(
                np.array(objectives), np.array(constraints)
            )

            net_xi = np.zeros(shape=Cs[0].shape)

            for i in range(len(Cs)):
                xi = Cs[i]
                net_xi += weights[i]*xi

            xi_A, xi_B = np.vsplit(net_xi, 2)
            net_cm = (nus[0]*xi_A + nus[1]*xi_B).reshape(2, 2)

            best_score = 1 - compute_gmean(net_cm)
            best_con = compute_EOpp([xi_A.reshape(2, 2), xi_B.reshape(2, 2)], pia)

            train_scores.append(best_score)
            train_cons_val.append(best_con)

            net_cm = np.zeros(shape = (2, 2))
            net_xi_A = np.zeros(shape = (2, 2))
            net_xi_B = np.zeros(shape = (2, 2))

            for i in range(len(Cs)):
                if weights[i] > 0:
                    clf = clfs[i]
                    L_A, L_B = clf
                    xi_A = get_confusion_matrix_from_loss(L_A, X_test[test_ind_a[0]], y_test[test_ind_a[0]], vec_eta, n_class)
                    xi_B = get_confusion_matrix_from_loss(L_B, X_test[test_ind_a[1]], y_test[test_ind_a[1]], vec_eta, n_class)
                    cm = (nus[0]*xi_A + nus[1]*xi_B).reshape(2, 2)
                    net_cm += weights[i]*cm
                    net_xi_A += weights[i]*xi_A
                    net_xi_B += weights[i]*xi_B

            # print(objectives)
            # print(constraints)

            score = 1 - compute_gmean(net_cm)
            con = compute_EOpp([net_xi_A.reshape(2, 2), net_xi_B.reshape(2, 2)], test_pia_a)

            test_scores.append(score)
            test_cons_val.append(con)

        print(train_scores)
        print(train_cons_val)

        print(test_scores)
        print(test_cons_val)

    np.save("./new-results/3pplug_" + name + "_train_scores.npy", train_scores)
    np.save("./new-results/3pplug_" + name + "_test_scores.npy", test_scores)
    np.save("./new-results/3pplug_" + name + "_train_cons.npy", train_cons_val)
    np.save("./new-results/3pplug_" + name + "_test_cons.npy", test_cons_val)