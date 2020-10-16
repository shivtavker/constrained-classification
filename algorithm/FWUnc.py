import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from models.logistic_regression import get_vec_eta_1
from models.fwunc import FWUnc_model
from standard_funcs.helpers import *
from standard_funcs.confusion_matrix import get_confusion_matrix_threshold
from standard_funcs.randomized_classifiers import get_confusion_matrix_final_loss

def FWUnc(X_train, y_train, X_test, y_test, loss_name, constraint_name, T):
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    vec_eta_1 = get_vec_eta_1(X_train, y_train)
    if(loss_name not in ["linear", "gmean", "hmean", "qmean"]):
        SystemExit("Unknown Loss")

    fwunc = FWUnc_model(X_train, y_train, vec_eta_1, loss_name)
    threshold = fwunc.run_algorithm(T)
    cm_train = get_confusion_matrix_threshold(threshold, X_train, y_train, vec_eta_1, "geq")
    cm_test = get_confusion_matrix_threshold(threshold, X_test, y_test, vec_eta_1, "geq")
    y_train_pred = 1*(vec_eta_1(X_train) > threshold)
    y_test_pred = 1*(vec_eta_1(X_test) > threshold)
    if(loss_name == "linear"):
        loss_train = compute_linear_loss_conf(cm_train)
        loss_test = compute_linear_loss_conf(cm_test)
    elif(loss_name == "gmean"):
        loss_train = compute_gmean_conf(cm_train, y_train)
        loss_test = compute_gmean_conf(cm_test, y_test)
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
    
    # print("Train Loss:", loss_train)
    # print("Train Constraint Value:", constraint_value_train)
    # print("Test Loss:", loss_test)
    # print("Test Constraint Value:", constraint_value_test)
    print(str(loss_train)[:5] + "(" + str(constraint_value_train)[:5] + ")")
    print(str(loss_test)[:5] + "(" + str(constraint_value_test)[:5] + ")")