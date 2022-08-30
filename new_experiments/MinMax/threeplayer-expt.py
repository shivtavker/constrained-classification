import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.threeplayer import threeplayer
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_minmax

### Data Loading
for d in ["pageblocks"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']
    train_res = [[0]*10]
    test_res = [[0]*10]

    T = 2000

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=0)

    if d == "MACHO":
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    ### Number of Classes
    n_class = len(np.unique(y_train))

    ### Training CPE Model
    vec_eta = get_vec_eta(X_train, y_train)

    ### Getting the Classifiers and Weights
    lr_range = [0.001, 0.01, 0.1]
    grid = [(xx, yy) for xx in lr_range for yy in lr_range]

    lr_perf = []

    for lr1, lr2 in grid:
        clfs, weights = threeplayer(X_train, y_train, vec_eta, T, n_class, lr1, lr2)

        ### Evaluate Performance on Train and Test Data
        train_conf = weight_confusion_matrix(X_train, y_train, clfs, weights, n_class, vec_eta)
        train_score = 1 - compute_minmax(train_conf)
        test_conf = weight_confusion_matrix(X_test, y_test, clfs, weights, n_class, vec_eta)
        test_score = 1 - compute_minmax(test_conf)

        lr_perf.append(train_score)
        # lr_perf.append(test_score)
    
    best_index = np.argmin(lr_perf)
    best_lr1, best_lr2 = grid[best_index]

    for T in list(np.arange(5, 500, 50)) + list(np.arange(500, 5000, 500)):
    # for T in [10, 50, 100, 1000, 2000, 3000, 5000]:
        train_scores = []
        test_scores = []

        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)

            if d == "MACHO":
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            ### Number of Classes
            n_class = len(np.unique(y_train))

            ### Training CPE Model
            vec_eta = get_vec_eta(X_train, y_train)

            ### Getting the Classifiers and Weights
            clfs, weights = threeplayer(X_train, y_train, vec_eta, T, n_class, best_lr1, best_lr2)

            ### Evaluate Performance on Train and Test Data
            train_conf = weight_confusion_matrix(X_train, y_train, clfs, weights, n_class, vec_eta)
            train_score = 1 - compute_minmax(train_conf)
            test_conf = weight_confusion_matrix(X_test, y_test, clfs, weights, n_class, vec_eta)
            test_score = 1 - compute_minmax(test_conf)

            train_scores.append(train_score)
            test_scores.append(test_score)

        mu_train = 1-round(np.mean(train_scores), 3)
        mu_test = 1-round(np.mean(test_scores), 3)
        std_train = round(2*np.std(train_scores)/np.sqrt(len(train_scores)), 3)
        std_test = round(2*np.std(test_scores)/np.sqrt(len(test_scores)), 3)
        
        # print(str(mu_train))
        # print(str(mu_test))
        # train_res.append(mu_train)
        # test_res.append(mu_test)

        print(str(mu_train) + " (" + str(std_train) + ")")
        print(str(mu_test) + " (" + str(std_test) + ")")

        train_res.append(train_scores)
        test_res.append(test_scores)

    np.save("./lmo-results-mod/threeplayerplg-" + d + "-train.npy", train_res)
    np.save("./lmo-results-mod/threeplayerplg-" + d + "-test.npy", test_res)

    # for s in train_res:
    #     print(str(s))
    # print("=====")
    # for s in test_res:
    #     print(str(s))
