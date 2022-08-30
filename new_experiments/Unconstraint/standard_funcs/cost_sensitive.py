import numpy as np
import tensorflow as tf
from standard_funcs.confusion_matrix import get_confusion_matrix

def get_confusion_matrix_from_loss_no_a(L_t, X_train, y_train, n_class):
    num_features = len(X_train[0])
    X_train = np.array(X_train, dtype="float32")
    L_t = np.array(L_t, dtype="float32")
    learning_rate = 1e-3
    training_steps = 200
    batch_size = 256
    display_step = 50

    train_data=tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data=train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

    W = tf.Variable(tf.ones([num_features, n_class]), name="weight")

    # Bias of shape [10], the total number of classes.

    b = tf.Variable(tf.zeros([n_class]), name="bias")
    optimizer = tf.optimizers.SGD(learning_rate)

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
    return get_confusion_matrix(y_train, y_pred, n_class)

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