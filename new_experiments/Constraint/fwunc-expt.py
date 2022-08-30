import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.fwunc import frank_wolfe_unc
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_hmean, compute_cov_value

### Data Loading
for d in ["MACHO"]:
# for d in ["glass"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    train_cons = []
    test_scores = []
    test_cons = []

    for global_i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=global_i)

        ### Number of Classes
        n_class = len(np.unique(y_train))

        ### Training CPE Model
        vec_eta = get_vec_eta(X_train, y_train)

        ### Getting the Classifiers and Weights
        clfs, weights = frank_wolfe_unc(X_train, y_train, vec_eta, 1000, n_class)

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

    mu_score_train = round(np.mean(train_scores), 3)
    mu_score_test = round(np.mean(test_scores), 3)
    std_score_train = round(np.std(train_scores), 3)
    std_score_test = round(np.std(test_scores), 3)

    mu_cons_train = round(np.mean(train_cons), 3)
    mu_cons_test = round(np.mean(test_cons), 3)
    std_cons_train = round(np.std(train_cons), 3)
    std_cons_test = round(np.std(test_cons), 3)

    print("===Hmean===")
    print(str(mu_score_train) + " (" + str(std_score_train) + ")")
    print(str(mu_score_test) + " (" + str(std_score_test) + ")")
    print("===Constraint===")
    print(str(mu_cons_train) + " (" + str(std_cons_train) + ")")
    print(str(mu_cons_test) + " (" + str(std_cons_test) + ")")

    np.save("fwunc_" + d + "_train_scores.npy", train_scores)
    np.save("fwunc_" + d + "_test_scores.npy", test_scores)

    np.save("fwunc_" + d + "_train_cons.npy", train_cons)
    np.save("fwunc_" + d + "_test_cons.npy", test_cons)
