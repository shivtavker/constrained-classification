import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.SBFW import SBFW

data = np.loadtxt("./data/adult.data", delimiter=",")
Y = data[:, 0]
X = data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=2)

eta_t_array = [0.5, 0.4, 0.3, 0.1]
lambda_val = 1
print("Dataset: Adult")
print(X_train.shape)
for loss_name in ["gmean", "hmean", "qmean"]:
    for param_cons in [("DP", 0.05), ("EOdds", 0.05), ("EOpp", 0.05), ("KLD", 0.01), ("COV", 0.25)]:
        constraint_name = param_cons[0]
        epsilon = param_cons[1]
        print(loss_name + " --- " + constraint_name + " < " + str(epsilon) + " ----- lambda = " + str(lambda_val) + " eta_t_array = " + str(eta_t_array))
        SBFW(X_train, y_train, X_test, y_test, loss_name, constraint_name, lambda_val, epsilon, eta_t_array, 500)
