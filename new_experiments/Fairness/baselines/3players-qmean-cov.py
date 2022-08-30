import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import math
import random
import numpy as np
from six.moves import xrange
from sklearn import linear_model
from sklearn import metrics
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco
import warnings

import logging
tf.get_logger().setLevel(logging.ERROR)

tf.disable_eager_execution()
warnings.filterwarnings('ignore')

for dataset_name in ["adult", "compas", "crimes", "default", "lawschool"]:

    ### Data Loading 
    data_dict = np.load("../data/" + dataset_name + "_data.npy", allow_pickle=True).item()
    x_train = data_dict.get('X_train')
    y_train = data_dict.get('y_train')
    z_train = x_train[:, 0]
    train_set = x_train, y_train, z_train

    x_test = data_dict['X_test']
    y_test = data_dict['y_test']
    z_test = x_test[:, 0]
    test_set = x_test, y_test, z_test

    epsilon = 0.25
    T = 2000

    ### Evaluation Functions
    def error_rate(features, labels, model_weights, threshold=0):
        predictions = np.dot(features, model_weights) + threshold
        y_pred = 1.0 * (predictions > 0)
        return np.mean(y_pred != labels)


    def expected_error_rate(features, labels, models, probabilities):
        er = 0.0
        for i in range(len(models)):
            er += probabilities[i] * error_rate(
                features, labels, models[i][0], models[i][1])
        # print(er)
        return er

    def pred_positive(features, labels, model_weights, threshold=0):
        predictions = np.dot(features, model_weights) + threshold
        y_pred = 1.0 * (predictions > 0)
        return np.mean(y_pred == 1)

    def expected_pred_positive(features, labels, models, probabilities):
        pred_pos = 0.0
        for i in range(len(models)):
            pred_pos += probabilities[i] * pred_positive(
                features, labels, models[i][0], models[i][1])
        # print(er)
        return pred_pos

    def qmean_error(features, labels, models, probabilities):
        tpr = 1 - expected_error_rate(
            features[labels == 1, :],  labels[labels == 1], models, probabilities)
        tnr = 1 - expected_error_rate(
            features[labels == 0, :],  labels[labels == 0], models, probabilities)
        
        return np.sqrt(((1-tpr)**2 + (1-tnr)**2)/2)


    def expected_fairness(features, labels, groups, models, probabilities):
        pp = expected_pred_positive(
            features, labels, models, 
            probabilities)
        return pp


    def evaluate_expected_results(dataset, model, probabilities):
        x, y, z = dataset
        error = qmean_error(x, y, model, probabilities)
        pp = expected_fairness(x, y, z, model, probabilities)
        return error, pp


    def print_results(train_set, test_set, stochastic_model, objectives=None, 
                    violations=None):
        x_train, y_train, z_train = train_set
        x_test, y_test, z_test = test_set

        models, probabilities = stochastic_model
        
        error_train, pp_train = evaluate_expected_results(
            train_set, models, probabilities)
        # print("Train qmean error = %.3f" % error_train)
        # print("Train fairness violation = %.3f" % abs(pp_train))
        # print()
        print("%.3f(%.3f)"% (error_train, abs(pp_train)))

        error_test, pp_test = evaluate_expected_results(
            test_set, models, probabilities)
        # print("Test qmean error = %.3f" % error_test)
        # print("Test fairness violation = %.3f" % abs(pp_test))
        print("%.3f(%.3f)"% (error_test, abs(pp_test)))

    ### Optimize Q-mean s.t. COV
    def lagrangian_optimizer(train_set, epsilon=epsilon, learning_rate=0.1, 
                            learning_rate_constraint=0.1, loops=2000):
        tf.reset_default_graph()
        
        x_train, y_train, z_train = train_set
        num_examples = x_train.shape[0]
        dimension = x_train.shape[-1]
        
        # Data tensors.
        features_tensor = tf.constant(x_train.astype("float32"), name="features")
        labels_tensor = tf.constant(y_train.astype("float32"), name="labels")

        # Linear model.
        weights = tf.Variable(tf.zeros(dimension, dtype=tf.float32), 
                                name="weights")
        threshold = tf.Variable(0, name="threshold", dtype=tf.float32)
        predictions_tensor = (tf.tensordot(features_tensor, weights, axes=(1, 0))
                                + threshold)

        predictions_group0 = tf.boolean_mask(predictions_tensor, mask=(z_train < 1))
        num0 = np.sum(z_train < 1)
        predictions_group1 = tf.boolean_mask(predictions_tensor, mask=(z_train > 0))
        num1 = np.sum(z_train > 0)

        # Set up rates.
        context = tfco.rate_context(predictions_tensor, labels_tensor)
        true_positive_rate = tfco.true_positive_rate(context)
        true_negative_rate = tfco.true_negative_rate(context)
        pred_positive = tfco.positive_prediction_rate(context)

        # Set up slack variables.
        slack_tpr = tf.Variable(0.5, dtype=tf.float32)
        slack_tnr = tf.Variable(0.5, dtype=tf.float32)
        
        # Projection ops for slacks. #Don't know
        projection_ops = []
        projection_ops.append(
            tf.assign(slack_tpr, tf.clip_by_value(slack_tpr, 0.001, 0.999)))
        projection_ops.append(
            tf.assign(slack_tnr, tf.clip_by_value(slack_tnr, 0.001, 0.999)))
        
        # Set up 1 - Q-mean objective.
        objective = tfco.wrap_rate(tf.sqrt(((1-slack_tpr)**2 + (1-slack_tnr)**2)/2))

        # Set up slack constraints.
        constraints = []
        constraints.append(tfco.wrap_rate(slack_tpr) <= true_positive_rate)
        constraints.append(tfco.wrap_rate(slack_tnr) <= true_negative_rate)

        # Set up COV constraints.
        constraints.append(pred_positive <= epsilon)

        # Set up constraint optimization problem.
        problem = tfco.RateMinimizationProblem(objective, constraints)

        # Set up solver.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        constraint_optimizer = tf.train.AdamOptimizer(learning_rate_constraint)
        lagrangian_optimizer = tfco.ProxyLagrangianOptimizerV1(
            optimizer=optimizer, constraint_optimizer=constraint_optimizer)
        train_op = lagrangian_optimizer.minimize(problem)

        # Start TF session and initialize variables.
        session = tf.Session()
        tf.set_random_seed(654321)  # Set random seed for reproducibility.
        session.run(tf.global_variables_initializer())

        # We maintain a list of objectives and model weights during training.
        objectives = []
        violations = []
        models = []
        
        # Perform  full gradient updates.
        for ii in xrange(loops):
            # Gradient update.
            session.run(train_op)
            # Projection.
            session.run(projection_ops)
            
            # Checkpoint once in 10 iterations.
            if ii % 100 == 0:
                # Model weights.
                model = [session.run(weights), session.run(threshold)]
                models.append(model)

                # Snapshot performace
                error, pp = evaluate_expected_results(
                    train_set, [model], [1.0])
                objectives.append(error)
                violations.append([pp - epsilon])

            # print("Step %d | Q-mean error = %3f | COV violation = %.3f" % (
            #     ii, objectives[-1], max(violations[-1])))
            
        # Use the recorded objectives and constraints to find the best iterate.
        # Best model
        best_iterate = tfco.find_best_candidate_index(
            np.array(objectives), np.array(violations))
        best_model = models[best_iterate]
        
        # Stochastic model over a subset of classifiers.
        probabilities = tfco.find_best_candidate_distribution(
            np.array(objectives), np.array(violations))
        models_pruned = [models[i] for i in range(len(models)) if probabilities[i] > 0.0]
        probabilities_pruned = probabilities[probabilities > 0.0]

        # Stochastic model over all classifiers.
        probabilities_all = probabilities * 0.0 + 1.0 / len(probabilities)
            
        # Return Pruned models, Avg models, Best model
        results = {
            'stochastic': (models, probabilities_all),
            'pruned': (models_pruned, probabilities_pruned),
            'best': ([best_model[0]], best_model[1]),
            'objectives': objectives,
            'violations': violations
        }
        return results

    lr_range = [0.01, 0.1, 1.0]
    grid = [(xx, yy) for xx in lr_range for yy in lr_range]
    objectives = []
    violations = []

    for (lr, lr_con) in grid:
        # print('Learning rate = %.3f | Constraint learning rate = %.3f' % (lr, lr_con))
        results = lagrangian_optimizer(
            train_set, epsilon=epsilon, learning_rate=lr, 
            learning_rate_constraint=lr_con)
        error, pp = evaluate_expected_results(
            train_set, results['stochastic'][0], results['stochastic'][1])
        objectives.append(error)
        violations.append([pp - epsilon])
        # print()

    best_index = tfco.find_best_candidate_index(
        np.array(objectives), np.array(violations), rank_objectives=False)
    # print('Retrain with learning rate (%.3f, %.3f)\n' % grid[best_index])
    results = lagrangian_optimizer(
            train_set, epsilon=epsilon, learning_rate=grid[best_index][0], 
            learning_rate_constraint=grid[best_index][1])
    print_results(
                train_set, test_set, results['stochastic'], results['objectives'], 
                results['violations'])

# print("Dataset: ", dataset_name)
# for learning_rate in [0.1, 0.5, 1]:
#     for learning_rate_constraint in [0.1, 0.5, 1]:
#         results = lagrangian_optimizer(train_set, learning_rate=learning_rate, learning_rate_constraint=learning_rate_constraint)
#         print("=========")
#         print("Learning Rate: " + str(learning_rate) + "  Rate Constraint: " + str(learning_rate_constraint))
#         print_results(
#             train_set, test_set, results['pruned'], results['objectives'], 
#             results['violations'])