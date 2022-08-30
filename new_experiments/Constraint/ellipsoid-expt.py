import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.ellipsoid import ellipsoid
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_hmean, compute_cov_value

### Data Loading
for d in ["abalone"]:
    t_train_scores = []
    t_train_cons = []
    t_test_scores = []
    t_test_cons = []

    eps = 0.01

    for T in [10, 50, 75, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]:
        data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
        X_data = data_dict['X']
        Y_data = data_dict['Y']
        train_scores = []
        train_cons = []
        test_scores = []
        test_cons = []

        for gloabal_i in range(1):
            # print(gloabal_i)
            X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=gloabal_i)

            ### Number of Classes
            n_class = len(np.unique(y_train))

            ### Training CPE Model
            vec_eta = get_vec_eta(X_train, y_train)
            
            ### Setting Target
            target = np.zeros((n_class,))
            for i in range(n_class):
                target[i] = (y_train == i).mean()

            ### Getting the Classifiers and Weights
            clfs, weights = ellipsoid(X_train, y_train, vec_eta, T, n_class, eps, target)

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

        print(np.median(train_scores))
        print(np.median(train_cons))
        print(np.median(test_scores))
        print(np.median(test_cons))
        
        t_train_scores.append(train_scores)
        t_train_cons.append(train_cons)
        t_test_scores.append(test_scores)
        t_test_cons.append(test_cons)
        print(T)

    # print(t_train_scores[-1])
    # print(t_train_cons[-1])
    # print(t_test_scores[-1])
    # print(t_test_cons[-1])

    np.save("./lmo-results/elp_" + d + "_train_scores.npy", np.array(t_train_scores))
    np.save("./lmo-results/elp_" + d + "_test_scores.npy", np.array(t_test_scores))

    np.save("./lmo-results/elp_" + d + "_train_cons.npy", np.array(t_train_cons))
    np.save("./lmo-results/elp_" + d + "_test_cons.npy", np.array(t_test_cons))

    #     mu_score_train = round(np.mean(train_scores), 3)
    #     mu_score_test = round(np.mean(test_scores), 3)
    #     std_score_train = round(np.std(train_scores), 3)
    #     std_score_test = round(np.std(test_scores), 3)

    #     mu_cons_train = round(np.mean(train_cons), 3)
    #     mu_cons_test = round(np.mean(test_cons), 3)
    #     std_cons_train = round(np.std(train_cons), 3)
    #     std_cons_test = round(np.std(test_cons), 3)

    # print("===Hmean===")
    # print(str(mu_score_train) + " (" + str(std_score_train) + ")")
    # print(str(mu_score_test) + " (" + str(std_score_test) + ")")
    # print("===Constraint===")
    # print(str(mu_cons_train) + " (" + str(std_cons_train) + ")")
    # print(str(mu_cons_test) + " (" + str(std_cons_test) + ")")