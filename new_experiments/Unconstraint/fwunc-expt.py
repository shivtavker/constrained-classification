import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fwunc import frank_wolfe_unc
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_hmean

### Data Loading
for d in ["MACHO"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    X_data = data_dict['X']
    Y_data = data_dict['Y']
    train_res = [[0]*10]
    test_res = [[0]*10]

    for T in [5000]:
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
            clfs, weights = frank_wolfe_unc(X_train, y_train, vec_eta, T, n_class)
            print(len(weights))
            print(len(clfs))
            ### Evaluate Performance on Train and Test Data
            train_conf = weight_confusion_matrix(X_train, y_train, clfs, weights, n_class, vec_eta)
            train_score = 1 - compute_hmean(train_conf)
            test_conf = weight_confusion_matrix(X_test, y_test, clfs, weights, n_class, vec_eta)
            test_score = 1 - compute_hmean(test_conf)

            train_scores.append(train_score)
            test_scores.append(test_score)

        mu_train = 1-round(np.mean(train_scores), 3)
        mu_test = 1-round(np.mean(test_scores), 3)
        std_train = round(2*np.std(train_scores)/np.sqrt(len(train_scores)), 3)
        std_test = round(2*np.std(test_scores)/np.sqrt(len(test_scores)), 3)
        
        print(str(mu_train) + " (" + str(std_train) + ")")
        print(str(mu_test) + " (" + str(std_test) + ")")
        # print(str(mu_train))
        # print(str(mu_test))
        # print("===")
        train_res.append(train_scores)
        test_res.append(test_scores)
    
    print(train_res)
    print(test_res)

    # np.save("./lmo-results-mod/fwunc-" + d + "-train.npy", train_res)
    # np.save("./lmo-results-mod/fwunc-" + d + "-test.npy", test_res)