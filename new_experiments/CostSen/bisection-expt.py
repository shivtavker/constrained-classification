import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.bisection import bisection
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_fmeasure

### Data Loading
for d in ["abalone", "adult", "glass", "pageblocks"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    test_scores = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)

        ### Number of Classes
        n_class = len(np.unique(y_train))

        ### Training CPE Model
        vec_eta = get_vec_eta(X_train, y_train)

        ### Getting the Classifiers and Weights
        clf = bisection(X_train, y_train, vec_eta, 1000, n_class)

        ### Evaluate Performance on Train and Test Data
        train_conf = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, n_class)
        train_score = compute_fmeasure(train_conf)
        test_conf = get_confusion_matrix_from_loss_no_a(clf, X_test, y_test, vec_eta, n_class)
        test_score = compute_fmeasure(test_conf)

        train_scores.append(train_score)
        test_scores.append(test_score)

    mu_train = round(np.mean(train_scores), 3)
    mu_test = round(np.mean(test_scores), 3)
    std_train = round(np.std(train_scores), 3)
    std_test = round(np.std(test_scores), 3)
    
    print(str(mu_train) + " (" + str(std_train) + ")")
    print(str(mu_test) + " (" + str(std_test) + ")")
