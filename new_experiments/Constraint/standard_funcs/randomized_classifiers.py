import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_a, get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import get_protected_indices, get_unique_a

def get_confusion_matrix_final_loss(clfs, weights, X_train, y_train, vec_eta_1, protected = True):
    clf_conf = []
    if(protected):
        unique_a = get_unique_a(X_train)
        protected_indices = get_protected_indices(X_train)
        for clf in clfs:
            clf_conf.append(get_confusion_matrix_from_loss_a(clf, X_train, y_train, unique_a, protected_indices, vec_eta_1))
        clf_conf = np.array(clf_conf)
        cm_final = np.average(clf_conf, axis=0, weights=weights[:len(clf_conf)])
    else:
        for clf in clfs:
            clf_conf.append(get_confusion_matrix_from_loss_no_a(clf, X_train, y_train, vec_eta_1))
        clf_conf = np.array(clf_conf)
        cm_final = np.average(clf_conf, axis=0, weights=weights[:len(clf_conf)])

    return cm_final