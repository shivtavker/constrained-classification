import numpy as np
from sklearn.model_selection import train_test_split
from models.logistic_regression import get_vec_eta_1

from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from standard_funcs.helpers import *

loss_name = "gmean"
constraint_name = "EOpp"
n_class = 2

for name in ["compas", "crimes", "default", "lawschool", "adult"]:
    data_dict = np.load("data/" + name +"_data.npy", allow_pickle=True).item()
    X_train = data_dict.get('X_train')
    y_train = data_dict.get('y_train')
    X_test = data_dict.get('X_test')
    y_test = data_dict.get('y_test')

    X = np.vstack((X_train, X_test))
    Y = np.hstack((y_train, y_test))

    train_scores = []
    test_scores = []
    train_cons_val = []
    test_cons_val = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=i, test_size=0.3)

        ### Training CPE Model
        vec_eta = get_vec_eta_1(X_train, y_train)

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
        cm_train, y_train_pred = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, True)
        cm_test, y_test_pred = get_confusion_matrix_from_loss_no_a(clf, X_test, y_test, vec_eta, True)

        if(loss_name == "linear"):
            loss_train = compute_linear_loss_conf(cm_train)
            loss_test = compute_linear_loss_conf(cm_test)
        elif(loss_name == "gmean"):
            loss_train = compute_gmean_conf(cm_train, y_train)
            loss_test = compute_gmean_conf(cm_test, y_test)
        elif(loss_name == "fmeasure"):
            loss_train = compute_fmeasure_conf(cm_train)
            loss_test = compute_fmeasure_conf(cm_test)
        elif(loss_name == "hmean"):
            loss_train = compute_hmean_conf(cm_train, y_train)
            loss_test = compute_hmean_conf(cm_test, y_test)
        elif(loss_name == "qmean"):
            loss_train = compute_qmean_conf(cm_train, y_train)
            loss_test = compute_qmean_conf(cm_test, y_test)
        if(constraint_name == "KLD"):
            constraint_value_train = compute_KLD_conf(cm_train, y_train)
            constraint_value_test = compute_KLD_conf(cm_test, y_test)
        elif(constraint_name == "COV"):
            constraint_value_train = compute_COV_conf(cm_train)
            constraint_value_test = compute_COV_conf(cm_test)
        if(constraint_name == "DP"):
            constraint_value_train = compute_DP(y_train, y_train_pred, X_train[:, 0])
            constraint_value_test = compute_DP(y_test, y_test_pred, X_test[:, 0])
        elif(constraint_name == "EOdds"):
            constraint_value_train = compute_EOdds(y_train, y_train_pred, X_train[:, 0])
            constraint_value_test = compute_EOdds(y_test, y_test_pred, X_test[:, 0])
        elif(constraint_name == "EOpp"):
            constraint_value_train = compute_EOpp(y_train, y_train_pred, X_train[:, 0])
            constraint_value_test = compute_EOpp(y_test, y_test_pred, X_test[:, 0])

        # # print("==LB==")
        # # print("Train Loss:", loss_train)
        # # print("Train Constraint Value:", constraint_value_train)
        # # print("Test Loss:", loss_test)
        # # print("Test Constraint Value:", constraint_value_test)
        # print(str(loss_train)[:5] + "(" + str(constraint_value_train)[:5] + ")")
        # print(str(loss_test)[:5] + "(" + str(constraint_value_test)[:5] + ")")

        train_scores.append(1 - loss_train)
        train_cons_val.append(constraint_value_train)
        test_scores.append(1 - loss_test)
        test_cons_val.append(constraint_value_test)

    np.save("logreg_" + name + "_train_scores.npy", train_scores)
    np.save("logreg_" + name + "_test_scores.npy", test_scores)

    np.save("logreg_" + name + "_train_cons.npy", train_cons_val)
    np.save("logreg_" + name + "_test_cons.npy", test_cons_val)

    # print(train_scores)
    # print(train_cons_val)
    # print(test_scores)
    # print(test_cons_val)

    # rf = RandomForestClassifier(max_leaf_nodes=10, n_estimators=100, class_weight="balanced").fit(X_train, y_train)

    # y_train_pred = rf.predict(X_train)
    # y_test_pred = rf.predict(X_test)
    # cm_train = get_confusion_matrix(y_train, y_train_pred)
    # cm_test = get_confusion_matrix(y_test, y_test_pred)

    # if(loss_name == "linear"):
    #     loss_train = compute_linear_loss_conf(cm_train)
    #     loss_test = compute_linear_loss_conf(cm_test)
    # elif(loss_name == "gmean"):
    #     loss_train = compute_gmean_conf(cm_train, y_train)
    #     loss_test = compute_gmean_conf(cm_test, y_test)
    # elif(loss_name == "hmean"):
    #     loss_train = compute_hmean_conf(cm_train, y_train)
    #     loss_test = compute_hmean_conf(cm_test, y_test)
    # elif(loss_name == "qmean"):
    #     loss_train = compute_qmean_conf(cm_train, y_train)
    #     loss_test = compute_qmean_conf(cm_test, y_test)
    # if(constraint_name == "KLD"):
    #     constraint_value_train = compute_KLD_conf(cm_train, y_train)
    #     constraint_value_test = compute_KLD_conf(cm_test, y_test)
    # elif(constraint_name == "COV"):
    #     constraint_value_train = compute_COV_conf(cm_train)
    #     constraint_value_test = compute_COV_conf(cm_test)
    # if(constraint_name == "DP"):
    #     constraint_value_train = compute_DP(y_train, y_train_pred, X_train[:, 0])
    #     constraint_value_test = compute_DP(y_test, y_test_pred, X_test[:, 0])
    # elif(constraint_name == "EOdds"):
    #     constraint_value_train = compute_EOdds(y_train, y_train_pred, X_train[:, 0])
    #     constraint_value_test = compute_EOdds(y_test, y_test_pred, X_test[:, 0])
    # elif(constraint_name == "EOpp"):
    #     constraint_value_train = compute_EOpp(y_train, y_train_pred, X_train[:, 0])
    #     constraint_value_test = compute_EOpp(y_test, y_test_pred, X_test[:, 0])

    # # print("==RF==")
    # # print("Train Loss:", loss_train)
    # # print("Train Constraint Value:", constraint_value_train)
    # # print("Test Loss:", loss_test)
    # # print("Test Constraint Value:", constraint_value_test)
    # print(str(loss_train)[:5] + "(" + str(constraint_value_train)[:5] + ")")
    # print(str(loss_test)[:5] + "(" + str(constraint_value_test)[:5] + ")")