import numpy as np
from algorithm.SBFW import SBFW

eta_t_array = [0.5, 0.4, 0.3, 0.1]
lambda_val = 1

data_dict = np.load("data/compas_data.npy").item()
X_train = data_dict.get('X_train')
y_train = data_dict.get('y_train')
X_test = data_dict.get('X_test')
y_test = data_dict.get('y_test')

SBFW(X_train, y_train, X_test, y_test, "gmean", "EOpp", lambda_val, 0.05, eta_t_array, 500)
