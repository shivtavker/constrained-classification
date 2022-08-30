import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from standard_funcs.confusion_matrix import get_confusion_matrix
from standard_funcs.helpers import compute_hmean

### Data Loading
for d in ["abalone", "adult", "glass", "pageblocks"]:
# for d in ["MACHO"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    test_scores = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        ### Number of Classes
        n_class = len(np.unique(y_train))

        ### Training LR Model

        lr = LogisticRegressionCV(solver="newton-cg", max_iter=100000, cv=2, tol=1e-3, multi_class="auto").fit(X_train, y_train)
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        ### Evaluate Performance on Train and Test Data
        train_conf = get_confusion_matrix(y_train, y_pred_train, n_class)
        train_score = 1 - compute_hmean(train_conf)
        test_conf = get_confusion_matrix(y_test, y_pred_test, n_class)
        test_score = 1 - compute_hmean(test_conf)

        train_scores.append(train_score)
        # print(train_score)
        test_scores.append(test_score)

    mu_train = round(np.mean(train_scores), 3)
    mu_test = round(np.mean(test_scores), 3)
    std_train = round(np.std(train_scores), 3)
    std_test = round(np.std(test_scores), 3)
    
    print(str(mu_train) + " (" + str(std_train) + ")")
    print(str(mu_test) + " (" + str(std_test) + ")")
