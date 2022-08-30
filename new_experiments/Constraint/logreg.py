import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_hmean, compute_fmeasure, compute_cov_value

### Data Loading
for d in ["MACHO"]:
# for d in ["MACHO"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    train_cons = []
    test_scores = []
    test_cons = []

    for global_i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=global_i)
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        ### Number of Classes
        n_class = len(np.unique(y_train))

        ### Training CPE Model
        vec_eta = get_vec_eta(X_train, y_train)

        ### Getting the Classifiers and Weights
        clf = np.zeros(shape=(n_class, n_class))

        ## Weights for Balance
        p = np.zeros((n_class,))
        for i in range(n_class):
            p[i] = (y_train == i).mean()

        for i in range(n_class):
            for j in range(n_class):
                if(i != j):
                    clf[i][j] = 1

        ### Evaluate Performance on Train and Test Data
        train_conf = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, n_class)
        train_score = 1 - compute_hmean(train_conf)
        test_conf = get_confusion_matrix_from_loss_no_a(clf, X_test, y_test, vec_eta, n_class)
        test_score = 1 - compute_hmean(test_conf)

        train_con = compute_cov_value(train_conf, p)

        p = np.zeros((n_class,))
        for i in range(n_class):
            p[i] = (y_test == i).mean()

        test_con = compute_cov_value(test_conf, p)

        train_scores.append(train_score)
        train_cons.append(train_con)

        # print(train_score)
        test_scores.append(test_score)
        test_cons.append(test_con)

    mu_train = round(np.mean(train_scores), 3)
    mu_test = round(np.mean(test_scores), 3)
    std_train = round(np.std(train_scores), 3)
    std_test = round(np.std(test_scores), 3)
    mu_tr_con =  round(np.mean(train_cons), 3)
    mu_te_con = round(np.mean(test_cons), 3)
    
    print(str(mu_train) + " (" + str(mu_tr_con) + ")")
    print(str(mu_test) + " (" + str(mu_te_con) + ")")

    np.save("logreg_" + d + "_train_scores.npy", train_scores)
    np.save("logreg_" + d + "_test_scores.npy", test_scores)

    np.save("logreg_" + d + "_train_cons.npy", train_cons)
    np.save("logreg_" + d + "_test_cons.npy", test_cons)
