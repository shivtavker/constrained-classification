import numpy as np
from algorithm.SBFW import SBFW
from sklearn.model_selection import train_test_split

## Eta Array to be used as decreasing step-function. Length automatically scales with T
eta_t_array = [0.5, 0.1, 0.01]
lambda_val = 10
epsilon = 0.025

for name in ["adult"]:
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
                loss_train, loss_test, constraint_value_train, constraint_value_test = SBFW(X_train, y_train, X_test, y_test, "gmean", "EOpp", lambda_val, epsilon, eta_t_array, 5000)

                train_scores.append(1 - loss_train)
                train_cons_val.append(constraint_value_train)
                test_scores.append(1 - loss_test)
                test_cons_val.append(constraint_value_test)

        np.save("./new-results/sbfw_" + name + "_train_scores.npy", train_scores)
        np.save("./new-results/sbfw_" + name + "_test_scores.npy", test_scores)

        np.save("./new-results/sbfw_" + name + "_train_cons.npy", train_cons_val)
        np.save("./new-results/sbfw_" + name + "_test_cons.npy", test_cons_val)

        print(train_scores)
        print(train_cons_val)
        print(test_scores)
        print(test_cons_val)