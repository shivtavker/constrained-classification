import numpy as np
# import tensorflow as tf
from standard_funcs.confusion_matrix import get_confusion_matrix
from sklearn.linear_model import LogisticRegression

def get_confusion_matrix_lr(L_t, X_train, y_train, n_class, X_test, y_test, first=None):
    first = None
    class_weights = -1*np.diagonal(L_t)
    class_weights[class_weights >= 0] += 1
    class_weights[class_weights < 0] = 0

    weights = {}
    for i in range(n_class):
        weights[i] = class_weights[i]

    lr_model = LogisticRegression(
        solver="newton-cg",
        max_iter=100000,
        # cv=2,
        tol=1e-3,
        multi_class="auto",
        class_weight=weights
            ).fit(X_train, y_train)
    
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)

    c_train = get_confusion_matrix(y_train, y_pred_train, n_class)
    c_test = get_confusion_matrix(y_test, y_pred_test, n_class)
    return (c_train, c_test)

def get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, n_class, X_test, y_test, first=True):
    num_features = len(X_train[0])
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    L_t = np.array(L_t, dtype="float32")
    learning_rate = 1e-5
    training_steps = 2000
    batch_size = len(X_train)
    display_step = 50

    train_data=tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data=train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

    if first:
        W = tf.Variable(tf.ones([num_features, n_class]), name="weight")
        b = tf.Variable(tf.zeros([n_class]), name="bias")
    else:
        W = tf.Variable(tf.convert_to_tensor(np.load('W.npy'), dtype=tf.float32), name="weight")
        b = tf.Variable(tf.convert_to_tensor(np.load('b.npy'), dtype=tf.float32), name="bias")

    optimizer = tf.optimizers.Adam(learning_rate)

    def run_optimization(x, y):
        with tf.GradientTape() as g:
            logits = logistic_regression(x)
            one_hots = tf.one_hot(y, n_class)
            loss = cost_weighted_crossentropy_loss(one_hots, logits, L_t)

        gradients = g.gradient(loss, [W, b])

        optimizer.apply_gradients(zip(gradients, [W, b]))

    def logistic_regression(x):
        return tf.matmul(x, W) + b

    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        run_optimization(batch_x, batch_y)
    
    probab = tf.nn.softmax(logistic_regression(X_train))
    y_pred = tf.argmax(probab, 1).numpy()
    c_train = get_confusion_matrix(y_train, y_pred, n_class)

    probab = tf.nn.softmax(logistic_regression(X_test))
    y_pred = tf.argmax(probab, 1).numpy()
    c_test = get_confusion_matrix(y_test, y_pred, n_class)

    np.save('W', W.numpy())
    np.save('b', b.numpy())

    return (c_train, c_test)

def cost_weighted_crossentropy_loss(onehot_labels, logits, cost_matrix):
    """Compute cost-weighted version of softmax cross-entropy loss.

    Args:
        onehot_labels: One-hot labels with shape (batch_size, num_classes)
        logits: Unnormalized prediction scores with shape (batch_size, num_classes)
        cost_matrix: Loss matrix with dimension (num_classes, num_classes)

    Returns:
        A `Tensor' of shape (batch_size,) with the per-example loss scores.
    """
    weights = tf.tensordot(onehot_labels, cost_matrix, axes=1)
    
    maximum_logits = tf.reduce_max(logits, axis=1, keepdims=True)
    numerators = tf.exp(logits - maximum_logits)
    denominators = tf.reduce_sum(numerators, axis=1, keepdims=True)
    log_probabilities = tf.math.log(numerators / denominators)

    weights = tf.cast(weights, dtype=tf.float32)
    maximum_weights = tf.reduce_max(weights, axis=1, keepdims=True)
    return (tf.squeeze(maximum_weights) - tf.reduce_sum((maximum_weights - weights) * (1 + log_probabilities), axis=1))