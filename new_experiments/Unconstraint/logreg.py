import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import compute_hmean, compute_fmeasure

### Data Loading
for d in ["abalone", "pageblocks", "MACHO", "satimage", "covtype"]:
# for d in ["MACHO"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    test_scores = []

    for i in range(10):
        u, count = np.unique(Y_data, return_counts=True)
        count_sort_ind = np.argsort(-count)
        classes = list(u[count_sort_ind])

        def func_map(x):
            return classes.index(x)

        Y_data = np.vectorize(func_map)(Y_data)

        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)
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

        for i in range(n_class):
            for j in range(n_class):
                if(i != j):
                    clf[i][j] = 1

        ### Evaluate Performance on Train and Test Data
        train_conf = get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta, n_class)
        train_score = 1 - compute_fmeasure(train_conf)
        test_conf = get_confusion_matrix_from_loss_no_a(clf, X_test, y_test, vec_eta, n_class)
        test_score = 1 - compute_fmeasure(test_conf)

        train_scores.append(train_score)
        # print(train_score)
        test_scores.append(test_score)

    mu_train = 1 - round(np.mean(train_scores), 3)
    mu_test = 1 - round(np.mean(test_scores), 3)
    std_train = round(2*np.std(train_scores)/np.sqrt(len(train_scores)), 3)
    std_test = round(2*np.std(test_scores)/np.sqrt(len(test_scores)), 3)
    
    print(str(mu_train) + " (" + str(std_train) + ")")
    print(str(mu_test) + " (" + str(std_test) + ")")
