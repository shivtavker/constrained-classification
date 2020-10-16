import numpy as np
from algorithm.SBFW import SBFW

## Eta Array to be used as decreasing step-function. Length automatically scales with T
eta_t_array = [0.5, 0.4, 0.3, 0.1, 1e-3]
lambda_val = 10

###Import Data here
data_dict = np.load("data/lawschool_data.npy", allow_pickle=True).item()
X_train = data_dict.get('X_train')
y_train = data_dict.get('y_train')
X_test = data_dict.get('X_test')
y_test = data_dict.get('y_test')

## Plugin Constraints
epsilon = 0.05
loss_train, loss_test, constraint_value_train, constraint_value_test = SBFW(X_train, y_train, X_test, y_test, "gmean", "EOpp", lambda_val, epsilon, eta_t_array, 2000)

print("Train Loss:", loss_train)
print("Train Constraint Violation:", constraint_value_train-epsilon)
print("Test Loss:", loss_test)
print("Test Constraint Violation:", constraint_value_test-epsilon)