import numpy as np

def get_confusion_matrix(Y_label, Y_pred):
    confusion_matrix = np.zeros(shape=(2, 2))
    np.add.at(confusion_matrix, (Y_label.astype(int), Y_pred.astype(int)), 1)
    return confusion_matrix/len(Y_label)

def get_confusion_matrix_threshold(threshold, X_train, y_train, vec_eta_1, inequality="leq"):
    confusion_matrix = np.zeros(shape=(2, 2))
    Y_pred = np.zeros(len(y_train))
    Pr_1 = vec_eta_1(X_train)
    if(inequality == "leq"):
        indices_1 = Pr_1 <= threshold
    else:
        indices_1 = Pr_1 >= threshold
    Y_pred[indices_1] = 1
    np.add.at(confusion_matrix, (y_train.astype(int), Y_pred.astype(int)), 1)
    return confusion_matrix/len(X_train)

def get_confusion_matrix_from_loss_a(L_t_array, X_train, y_train, unique_a, protected_indices, vec_eta_1):
    confusion_matrices = []
    
    for i in range(unique_a):
        L_t_a = L_t_array[i]
        confusion_matrix_a = np.zeros(shape=(2, 2))
        Y_pred = np.zeros(len(y_train[protected_indices[i]]))
        Pr_1 = vec_eta_1(X_train[protected_indices[i]])
        Pr_0 = 1 - Pr_1
        Loss_1 = L_t_a[1, 1]*Pr_1 + L_t_a[0, 1]*Pr_0
        Loss_0 = L_t_a[0, 0]*Pr_0 + L_t_a[1, 0]*Pr_1
        indices_1 = Loss_1 <= Loss_0
        Y_pred[indices_1] = 1
        np.add.at(confusion_matrix_a, (y_train[protected_indices[i]].astype(int), Y_pred.astype(int)), 1)
        confusion_matrix_a = confusion_matrix_a/len(Y_pred)
        confusion_matrices.append(confusion_matrix_a)
    
    return np.array(confusion_matrices)

def get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta_1, need_pred=False):
    confusion_matrix = np.zeros(shape=(2, 2))
    Y_pred = np.zeros(len(y_train))
    Pr_1 = vec_eta_1(X_train)
    Pr_0 = 1 - Pr_1
    Loss_1 = L_t[1, 1]*Pr_1 + L_t[0, 1]*Pr_0
    Loss_0 = L_t[0, 0]*Pr_0 + L_t[1, 0]*Pr_1
    
    indices_1 = Loss_1 <= Loss_0
    Y_pred[indices_1] = 1
    np.add.at(confusion_matrix, (y_train.astype(int), Y_pred.astype(int)), 1)

    if need_pred:
        return (confusion_matrix/len(X_train), Y_pred)

    return confusion_matrix/len(X_train)