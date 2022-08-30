import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm.fwunc import frank_wolfe_unc
from models.logistic_regression import get_vec_eta
from standard_funcs.confusion_matrix import weight_confusion_matrix
from standard_funcs.helpers import compute_hmean
from sklearn.metrics import confusion_matrix, f1_score

### Data Loading
for d in ["abalone", "pageblocks", "MACHO", "satimage", "covtype"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()
    # print(data_dict.keys())
    X_data = data_dict['X']
    Y_data = data_dict['Y']
    train_res = [0]*10
    test_res = [0]*10

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)
        # X_train = data_dict['X_train']
        # X_test = data_dict['X_test']
        # y_train = data_dict['y_train']
        # y_test = data_dict['y_test']

        n_class = len(np.unique(y_train))

        y_train_pred = np.random.randint(n_class, size=len(y_train))
        y_test_pred = np.random.randint(n_class, size=len(y_test))

        train_res[i] = compute_hmean(confusion_matrix(y_train, y_train_pred))
        test_res[i] = compute_hmean(confusion_matrix(y_test, y_test_pred))
    
    print(np.round(np.mean(train_res), decimals=3))
    print(np.round(np.mean(test_res), decimals=3))


    # np.save("./lmo-results-mod/fwunc-" + d + "-train.npy", train_res)
    # np.save("./lmo-results-mod/fwunc-" + d + "-test.npy", test_res)
