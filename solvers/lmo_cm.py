from standard_funcs.confusion_matrix import get_confusion_matrix_from_loss_a, get_confusion_matrix_from_loss_no_a
from standard_funcs.helpers import get_protected_indices

def LMO_CM_protected(grad_matrix, X_train, y_train, unique_a, vec_eta_1):
    """
    LMO for set of achievable Confusion Matrix with protected attribute

    Args:
        grad_matrix : shape n x n x unique_a
        unique_a: int
        epsilon: int

    Returns:
        (confusion_matrix, Loss_matrix to store clfs)

    """
    protected_indices = get_protected_indices(X_train)

    return (get_confusion_matrix_from_loss_a(grad_matrix, X_train, y_train, unique_a, protected_indices, vec_eta_1), grad_matrix)

def LMO_CM_no_protected(grad_matrix, X_train, y_train, vec_eta_1):
    """
    LMO for set of achievable Confusion Matrix without protected attribute

    Args:
        grad_matrix : shape n x n x unique_a
        unique_a: int
        epsilon: int

    Returns:
        (confusion_matrix, Loss_matrix to store clfs)

    """
    return (get_confusion_matrix_from_loss_no_a(grad_matrix, X_train, y_train, vec_eta_1), grad_matrix)

def LMO_CM_threshold(grad_matrix, X_train, y_train, vec_eta_1):
    """
    LMO for set of achievable Confusion Matrix without protected attribute

    Args:
        grad_matrix : shape n x n x unique_a
        unique_a: int
        epsilon: int

    Returns:
        (confusion_matrix, Loss_matrix to store clfs)

    """
    L_t = grad_matrix
    threshold = (L_t[0,0] - L_t[0,1])/(L_t[0,0] + L_t[1,1] - L_t[0,1] - L_t[1,0])
    return (get_confusion_matrix_from_loss_no_a(grad_matrix, X_train, y_train, vec_eta_1), threshold)