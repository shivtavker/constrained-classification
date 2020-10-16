import numpy as np
from sklearn.model_selection import train_test_split
from algorithm.SBFW import SBFW

eta_t_array = [0.5, 0.4, 0.3, 0.1]
lambda_val = 1
epsilon = 0.05 ## Slack available -- Lower values are preffered but takes more iterations

data_dict = np.load("data/adult_data.npy", allow_pickle=True).item()
X_train = data_dict.get('X_train')
y_train = data_dict.get('y_train')
X_test = data_dict.get('X_test')
y_test = data_dict.get('y_test')

loss_train, loss_test, constraint_value_train, constraint_value_test = SBFW(X_train, y_train, X_test, y_test, "gmean", "EOpp", lambda_val, epsilon, eta_t_array, 500)

print("Train Loss:", round(loss_train, 3))
print("Train Constraint Violation:", round(max(constraint_value_train-epsilon, 0), 3))
print("Test Loss:", round(loss_test, 3))
print("Test Constraint Violation:", round(max(constraint_value_test-epsilon, 0), 3))