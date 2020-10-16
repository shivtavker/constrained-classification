import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from standard_funcs.helpers import compute_qmean

def get_vec_eta_1(X_train, y_train):
    # params_tune = [(0.01, "l1"), (1, "l1"), (10, "l1"), (0.01, "l2"), (1, "l2"), (10, "l2")]
    params_tune = [(1, "l1"), (1, "l2")]
    accuracy_tries = np.zeros(len(params_tune))
    for i in range(len(params_tune)):
        # C_param = params_tune[i][0]
        penalty_param = params_tune[i][1]
        lr = LogisticRegressionCV(solver="liblinear", 
                                max_iter=1000,
                                # C=C_param,
                                penalty=penalty_param,
                                cv=2
                                ).fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        accuracy_tries[i] = accuracy_score(y_train, y_pred)
        
    best_param = params_tune[np.argmax(accuracy_tries)]

    lr = LogisticRegressionCV(solver="liblinear", 
                                max_iter=1000, 
                                # C=best_param[0], 
                                penalty=best_param[1],
                                cv=2
                                ).fit(X_train, y_train)

    def vec_eta_1(X):
        return lr.predict_proba(X)[:, 1]

    return vec_eta_1
    # y_pred = lr.predict(X_test)
    # print("Test Q mean Loss: ", compute_qmean(y_test, y_pred))
    # print("Test Demographic Parity: ", compute_demographic_parity(y_test, y_pred, X_test[:, 0]))

    # y_pred = lr.predict(X_train)
    # print("Train Q mean Loss: ", compute_qmean(y_train, y_pred))
    # print("Train Demographic Parity: ", compute_demographic_parity(y_train, y_pred, X_train[:, 0]))