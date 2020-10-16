import numpy as np
from algorithm.SBFW import SBFW

## Eta Array to be used as decreasing step-function. Length automatically scales with T
eta_t_array = [0.5, 0.1, 1e-3]
lambda_val = 10
for loss, cons, epsilon in [("gmean", "EOpp", 0.05), ("hmean", "COV", 0.25), ("qmean", "DP", 0.05)]:
    print("Optimizing " + loss.title() + " s.t. " + cons + " <= " + str(epsilon))
    for name in ["adult", "compas", "crimes", "default", "lawschool"]:
        print("=== Dataset:", name.upper() + " ===")
        data_dict = np.load("data/" + name +"_data.npy", allow_pickle=True).item()
        X_train = data_dict.get('X_train')
        y_train = data_dict.get('y_train')
        X_test = data_dict.get('X_test')
        y_test = data_dict.get('y_test')

        loss_train, loss_test, constraint_value_train, constraint_value_test = SBFW(X_train, y_train, X_test, y_test, loss, cons, lambda_val, epsilon, eta_t_array, 2000)

        print("Train Loss:", round(loss_train, 3))
        print("Train Constraint Violation:", round(max(constraint_value_train-epsilon, 0), 3))
        print("Test Loss:", round(loss_test, 3))
        print("Test Constraint Violation:", round(max(constraint_value_test-epsilon, 0), 3))