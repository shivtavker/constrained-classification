import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from models.logistic_regression import get_vec_eta_1
from models.sbfw import SBFW_protected, SBFW_no_protected
from standard_funcs.helpers import *
from standard_funcs.randomized_classifiers import get_confusion_matrix_final_loss

def SBFW(X_train, y_train, X_test, y_test, loss_name, constraint_name, lambda_val=1, epsilon=0.05, eta_t_array=[0.5, 0.4, 0.3, 0.1], T=500, alt_loss=False):
    y_train.astype(int)
    y_test.astype(int)
    vec_eta_1 = get_vec_eta_1(X_train, y_train)
    protected_constraint = ["DP", "EOdds", "EOpp"]
    not_protected_constraint = ["KLD", "COV"]
    protected = True
    if(constraint_name in protected_constraint):
        protected = True
    elif(constraint_name in not_protected_constraint):
        protected = False
    else:
        SystemExit("Unknown Constraint")
    if(loss_name not in ["linear", "gmean", "hmean", "qmean"]):
        SystemExit("Unknown Loss")

    if(protected):
        sbfw = SBFW_protected(X_train, y_train, vec_eta_1, loss_name, constraint_name, lambda_val, epsilon)
        clfs, weights = sbfw.run_algo(eta_t_array, T)
        final_cm_train = get_confusion_matrix_final_loss(clfs, weights, X_train, y_train, vec_eta_1, protected) 
        final_cm_test = get_confusion_matrix_final_loss(clfs, weights, X_test, y_test, vec_eta_1, protected)
        if(loss_name == "linear"):
            if(alt_loss == False):
                loss_train = compute_linear_loss_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)))
                loss_test = compute_linear_loss_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)))
            elif(alt_loss == "gmean"):
                loss_train = compute_gmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)), y_train)
                loss_test = compute_gmean_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)), y_test)
            elif(alt_loss == "hmean"):
                loss_train = compute_hmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)), y_train)
                loss_test = compute_hmean_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)), y_test)
            elif(alt_loss == "qmean"):
                loss_train = compute_qmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)), y_train)
                loss_test = compute_qmean_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)), y_test)
        elif(loss_name == "gmean"):
            loss_train = compute_gmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)), y_train)
            loss_test = compute_gmean_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)), y_test)
        elif(loss_name == "hmean"):
            loss_train = compute_hmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)), y_train)
            loss_test = compute_hmean_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)), y_test)
        elif(loss_name == "qmean"):
            loss_train = compute_qmean_conf(np.average(final_cm_train, axis=0, weights=get_len_protected(X_train)/len(y_train)), y_train)
            loss_test = compute_qmean_conf(np.average(final_cm_test, axis=0, weights=get_len_protected(X_test)/len(y_test)), y_test)
        if(constraint_name == "DP"):
            constraint_value_train = compute_DP_cms(final_cm_train, get_unique_a(X_train))
            constraint_value_test = compute_DP_cms(final_cm_test, get_unique_a(X_test))
        elif(constraint_name == "EOdds"):
            constraint_value_train = compute_EOdds_cms(final_cm_train, get_unique_a(X_train))
            constraint_value_test = compute_EOdds_cms(final_cm_test, get_unique_a(X_test))
        elif(constraint_name == "EOpp"):
            constraint_value_train = compute_EOpp_cms(final_cm_train, get_unique_a(X_train))
            constraint_value_test = compute_EOpp_cms(final_cm_test, get_unique_a(X_test))
    else:
        sbfw = SBFW_no_protected(X_train, y_train, vec_eta_1, loss_name, constraint_name, lambda_val, epsilon)
        clfs, weights = sbfw.run_algo(eta_t_array, T)
        final_cm_train = get_confusion_matrix_final_loss(clfs, weights, X_train, y_train, vec_eta_1, protected) 
        final_cm_test = get_confusion_matrix_final_loss(clfs, weights, X_test, y_test, vec_eta_1, protected)
        if(loss_name == "linear"):
            if(alt_loss == False):
                loss_train = compute_linear_loss_conf(final_cm_train)
                loss_test = compute_linear_loss_conf(final_cm_test)
            elif(alt_loss == "gmean"):
                loss_train = compute_gmean_conf(final_cm_train, y_train)
                loss_test = compute_gmean_conf(final_cm_test, y_test)
            elif(alt_loss == "hmean"):
                loss_train = compute_hmean_conf(final_cm_train, y_train)
                loss_test = compute_hmean_conf(final_cm_test, y_test)
            elif(alt_loss == "qmean"):
                loss_train = compute_qmean_conf(final_cm_train, y_train)
                loss_test = compute_qmean_conf(final_cm_test, y_test)
        elif(loss_name == "gmean"):
            loss_train = compute_gmean_conf(final_cm_train, y_train)
            loss_test = compute_gmean_conf(final_cm_test, y_test)
        elif(loss_name == "hmean"):
            loss_train = compute_hmean_conf(final_cm_train, y_train)
            loss_test = compute_hmean_conf(final_cm_test, y_test)
        elif(loss_name == "qmean"):
            loss_train = compute_qmean_conf(final_cm_train, y_train)
            loss_test = compute_qmean_conf(final_cm_test, y_test)
        if(constraint_name == "KLD"):
            constraint_value_train = compute_KLD_conf(final_cm_train, y_train)
            constraint_value_test = compute_KLD_conf(final_cm_test, y_test)
        elif(constraint_name == "COV"):
            constraint_value_train = compute_COV_conf(final_cm_train)
            constraint_value_test = compute_COV_conf(final_cm_test)
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
    # print(str(loss_train)[:5] + "(" + str(constraint_value_train)[:5] + ")")
    # print(str(loss_test)[:5] + "(" + str(constraint_value_test)[:5] + ")")
    # return (round(loss_train, 3), round(constraint_value_train, 3))
    return [loss_train, loss_test, constraint_value_train, constraint_value_test]