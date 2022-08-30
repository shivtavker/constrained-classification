import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_no_a, get_confusion_vector
from standard_funcs.helpers import get_protected_indices

def LMO_CM(grad_vector, X_train, y_train, vec_eta, n_class):
    L_t = np.ones(shape=(n_class, n_class))
    
    for i in range(n_class):
        L_t[:, i] = grad_vector[i+n_class]*L_t[:, i]
    
    for i in range(n_class):
        L_t[i, i] = grad_vector[i]
    
    cm = get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, vec_eta, n_class)
    
    return (get_confusion_vector(cm), L_t)