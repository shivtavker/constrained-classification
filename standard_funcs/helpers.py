import numpy as np
from standard_funcs.confusion_matrix import get_confusion_matrix

### Gives array of zeros with particular indices as 1
def get_row(array_indices, modify_indices, unique_a):
    rows = []
    for i in range(unique_a):
        rows.append(np.zeros(4))
    
    for index in array_indices:
        for j in modify_indices:
            rows[index][j] = 1
    
    return np.array(rows).flatten()

def get_unique_a(X_train):
    return len(np.unique(X_train[:, 0]))

def get_protected_indices(X_train):
    protected_indices = []
    unique_a = get_unique_a(X_train)
    len_protected = np.zeros(unique_a)

    for i in range(unique_a):
        protected_indices_a = X_train[:, 0] == i
        protected_indices.append(protected_indices_a)
        len_protected_a = np.count_nonzero(protected_indices_a)
        len_protected[i] = len_protected_a
    return protected_indices

def get_len_protected(X_train):
    unique_a = get_unique_a(X_train)
    len_protected = np.zeros(unique_a)
    protected_indices = get_protected_indices(X_train)
    for i in range(unique_a):
        protected_indices_a = protected_indices[i]
        len_protected_a = np.count_nonzero(protected_indices_a)
        len_protected[i] = len_protected_a
    return len_protected

def compute_linear_loss(Y_label, Y_pred):
    confusion_matrix = get_confusion_matrix(Y_label, Y_pred)
    return 1 - confusion_matrix[0][0] - confusion_matrix[1][1]

def compute_gmean(Y_label, Y_pred):
    pi_array = []
    for i in range(2):
        pi_a = np.count_nonzero(Y_label == i)/len(Y_label)
        pi_array.append(pi_a)
    
    confusion_matrix = get_confusion_matrix(Y_label, Y_pred)
    prod_form = 1
    for i in range(2):
        prod_form *= confusion_matrix[i][i]/pi_array[i]
    return 1 - (prod_form)**(1/2)

def compute_hmean(Y_label, Y_pred):
    pi_array = []
    for i in range(2):
        pi_a = np.count_nonzero(Y_label == i)/len(Y_label)
        pi_array.append(pi_a)
    
    confusion_matrix = get_confusion_matrix(Y_label, Y_pred)
    sum_formulae = 0
    for i in range(2):
        sum_formulae += pi_array[i]/confusion_matrix[i][i]
    return 1 - (2/sum_formulae)

def compute_qmean(Y_label, Y_pred):
    pi_array = []
    for i in range(2):
        pi_a = np.count_nonzero(Y_label == i)/len(Y_label)
        pi_array.append(pi_a)
    
    confusion_matrix = get_confusion_matrix(Y_label, Y_pred)
    sum_sq_form = 0
    for i in range(2):
        sum_sq_form += (1 - (confusion_matrix[i][i]/pi_array[i]))**2
    return np.sqrt(sum_sq_form/2)

def compute_linear_loss_conf(confusion_matrix):
    return 1 - confusion_matrix[0][0] - confusion_matrix[1][1]

def compute_gmean_conf(confusion_matrix, Y_label):
    pi_0 = np.count_nonzero(Y_label == 0)/len(Y_label)
    pi_1 = 1 - pi_0
    return 1 - np.sqrt(confusion_matrix[0][0] * confusion_matrix[1][1] / (pi_0*pi_1))

def compute_hmean_conf(confusion_matrix, Y_label):
    pi_0 = np.count_nonzero(Y_label == 0)/len(Y_label)
    pi_1 = 1 - pi_0
    return 1 - (2/((pi_0/confusion_matrix[0][0]) + (pi_1/confusion_matrix[1][1])))

def compute_qmean_conf(confusion_matrix, Y_label):
    pi_0 = np.count_nonzero(Y_label == 0)/len(Y_label)
    pi_1 = 1 - pi_0
    return np.sqrt(0.5*((1 - confusion_matrix[0][0]/pi_0)**2 + (1 - confusion_matrix[1][1]/pi_1)**2))

def compute_DP(Y_label, Y_pred, X_pro_feature):
    final_conf = np.zeros(shape=(2, 2))
    cm = []
    unique_a = len(np.unique(X_pro_feature))
    
    for a in np.unique(X_pro_feature).astype(int):
        indices_a = X_pro_feature == a
        Y_label_a = Y_label[indices_a]
        Y_pred_a = Y_pred[indices_a]
        cm_a = get_confusion_matrix(Y_label_a, Y_pred_a)
        final_conf += cm_a
        cm.append(cm_a)
    
    parity_per_a = np.zeros(unique_a)
    
    for i in range(unique_a):
        cm_a = cm[i]
        parity_per_a[i] = abs(cm_a[0, 1] + cm_a[1, 1] - (1/unique_a)*(final_conf[0, 1] + final_conf[1, 1]))
    
    return max(parity_per_a)

def compute_EOdds(Y_label, Y_pred, X_pro_feature):
    n_class = 2
    final_conf = np.zeros(shape=(n_class, n_class))
    cm = []
    unique_a = len(np.unique(X_pro_feature))
    
    for a in np.unique(X_pro_feature).astype(int):
        indices_a = X_pro_feature == a
        Y_label_a = Y_label[indices_a]
        Y_pred_a = Y_pred[indices_a]
        cm_a = get_confusion_matrix(Y_label_a, Y_pred_a)
        final_conf += cm_a
        cm.append(cm_a)
    
    odds_per_a_0 = np.zeros(unique_a)
    odds_per_a_1 = np.zeros(unique_a)
    
    for i in range(unique_a):
        cm_a = cm[i]
        odds_per_a_0[i] = abs(cm_a[0, 1] - (1/unique_a)*(final_conf[0, 1]))
        odds_per_a_1[i] = abs(cm_a[1, 1] - (1/unique_a)*(final_conf[1, 1]))
    
    return max(max(odds_per_a_0), max(odds_per_a_1))

def compute_KLD(Y_label, Y_pred):
    n_class = 2
    cm = get_confusion_matrix(Y_label, Y_pred)
    pi_array = []
    for i in range(n_class):
        pi_a = np.count_nonzero(Y_label == i)
        pi_array.append(pi_a/len(Y_label))

    kld_error = 0
    for i in range(n_class):
        sum_col_i = 0
        for j in range(n_class):
            sum_col_i += cm[j, i]
        kld_error += pi_array[i]*np.log(pi_array[i]/sum_col_i)
    
    return kld_error

def compute_COV(Y_label, Y_pred):
    n_class = 2
    cm = get_confusion_matrix(Y_label, Y_pred)
    return cm[0, 1] + cm[1, 1]

def compute_EOpp(Y_label, Y_pred, X_pro_feature):
    n_class = 2
    final_conf = np.zeros(shape=(n_class, n_class))
    cm = []
    unique_a = len(np.unique(X_pro_feature))
    
    for a in np.unique(X_pro_feature).astype(int):
        indices_a = X_pro_feature == a
        Y_label_a = Y_label[indices_a]
        Y_pred_a = Y_pred[indices_a]
        cm_a = get_confusion_matrix(Y_label_a, Y_pred_a)
        cm_a = cm_a/(cm_a[1, 0] + cm_a[1, 1])
        final_conf += cm_a
        cm.append(cm_a)
    odds_per_a_1 = np.zeros(unique_a)
    
    for i in range(unique_a):
        cm_a = cm[i]
        odds_per_a_1[i] = abs(cm_a[1, 1] - (1/unique_a)*(final_conf[1, 1]))
    
    return max(odds_per_a_1)

def compute_DP_cms(cms, unique_a):
    final_conf = np.sum(cms, axis=0)
    parity_per_a = np.zeros(unique_a)
    
    for i in range(unique_a):
        cm_a = cms[i]
        parity_per_a[i] = abs(cm_a[0, 1] + cm_a[1, 1] - (1/unique_a)*(final_conf[0, 1] + final_conf[1, 1]))
    
    return max(parity_per_a)

def compute_EOdds_cms(cms, unique_a):
    final_conf = np.sum(cms, axis=0)
    odds_per_a_0 = np.zeros(unique_a)
    odds_per_a_1 = np.zeros(unique_a)
    
    for i in range(unique_a):
        cm_a = cms[i]
        odds_per_a_0[i] = abs(cm_a[0, 1] - (1/unique_a)*(final_conf[0, 1]))
        odds_per_a_1[i] = abs(cm_a[1, 1] - (1/unique_a)*(final_conf[1, 1]))
    
    return max(max(odds_per_a_0), max(odds_per_a_1))

def compute_EOpp_cms(cms, unique_a):
    for i in range(unique_a):
        cms[i] = cms[i]/(cms[i][1, 0] + cms[i][1, 1])
    final_conf = np.sum(cms, axis=0)
    odds_per_a_1 = np.zeros(unique_a)
    
    for i in range(unique_a):
        cm_a = cms[i]
        odds_per_a_1[i] = abs(cm_a[1, 1] - (1/unique_a)*(final_conf[1, 1]))
    
    return max(odds_per_a_1)

def compute_KLD_conf(confusion_matrix, Y_label):
    pi_array = []
    pi_array.append(np.count_nonzero(Y_label == 0)/len(Y_label))
    pi_array.append(1 - pi_array[-1])
    kld_error = 0
    for i in range(2):
        sum_col_i = 0
        for j in range(2):
            sum_col_i += confusion_matrix[j, i]
        kld_error += pi_array[i]*np.log(pi_array[i]/sum_col_i)
    
    return kld_error

def compute_COV_conf(confusion_matrix):
    return confusion_matrix[0, 1] + confusion_matrix[1, 1]

