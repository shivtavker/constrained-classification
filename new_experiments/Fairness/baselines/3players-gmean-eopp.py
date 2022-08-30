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
from sklearn.model_selection import train_test_split


import logging
tf.get_logger().setLevel(logging.ERROR)

tf.disable_eager_execution()
warnings.filterwarnings('ignore')

for dataset_name in ["compas", "crimes", "default", "lawschool", "adult"]:

    train_scores = []
    test_scores = []
    train_cons_val = []
    test_cons_val = []

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

    X = np.vstack((x_train, x_test))
    Y = np.hstack((y_train, y_test))

    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=i, test_size=0.3)

        z_train = x_train[:, 0]
        train_set = x_train, y_train, z_train
        z_test = x_test[:, 0]
        test_set = x_test, y_test, z_test

        epsilon = 0.01

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


        def gmean_error(features, labels, models, probabilities):
            tpr = 1 - expected_error_rate(
                features[labels == 1, :],  labels[labels == 1], models, probabilities)
            tnr = 1 - expected_error_rate(
                features[labels == 0, :],  labels[labels == 0], models, probabilities)
            
            return 1 - np.sqrt(tpr * tnr)


        def expected_fairness(features, labels, groups, models, probabilities):
            pos_features = features[labels == 1, :]
            pos_labels = labels[labels == 1]
            pos_groups = groups[labels == 1]

            tpr_0 = 1 - expected_error_rate(
                pos_features[pos_groups == 0, :], pos_labels[pos_groups == 0], models, 
                probabilities)
            tpr_1 = 1 - expected_error_rate(
                pos_features[pos_groups == 1, :], pos_labels[pos_groups == 1], models, 
                probabilities)
            
            return tpr_0, tpr_1


        def evaluate_expected_results(dataset, model, probabilities):
            x, y, z = dataset
            error = gmean_error(x, y, model, probabilities)
            tpr0, tpr1 = expected_fairness(x, y, z, model, probabilities)
            return error, tpr0, tpr1


        def print_results(train_set, test_set, stochastic_model, objectives=None, 
                        violations=None):
            x_train, y_train, z_train = train_set
            x_test, y_test, z_test = test_set

            models, probabilities = stochastic_model
            
            error_train, tpr0_train, tpr1_train = evaluate_expected_results(
                train_set, models, probabilities)
            #   print("Train gmean error = %.3f" % error_train)
            #   print("Train fairness violation = %.3f (%.3f, %.3f)" % 
            #         (abs(tpr0_train - tpr1_train), tpr0_train, tpr1_train))
            #   print()
            # print("%.3f(%.3f)"% (error_train, abs(tpr0_train - tpr1_train)))

            train_scores.append(1 - error_train)
            train_cons_val.append(abs(tpr0_train - tpr1_train))

            error_test, tpr0_test, tpr1_test = evaluate_expected_results(
                test_set, models, probabilities)
            # print("%.3f(%.3f)"% (error_test, abs(tpr0_test - tpr1_test)))

            test_scores.append(1 - error_test)
            test_cons_val.append(abs(tpr0_test - tpr1_test))

        ### Optimize G-mean s.t. EOPP
        def lagrangian_optimizer(train_set, epsilon=epsilon, learning_rate=0.01, 
                                learning_rate_constraint=0.01, loops=2000):
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

            context0 = context.subset(z_train < 1)
            true_positive_rate0 = tfco.true_positive_rate(context0)

            context1 = context.subset(z_train > 0)
            true_positive_rate1 = tfco.true_positive_rate(context1)

            # Set up slack variables.
            slack_tpr = tf.Variable(0.5, dtype=tf.float32)
            slack_tnr = tf.Variable(0.5, dtype=tf.float32)
            
            # Projection ops for slacks.
            projection_ops = []
            projection_ops.append(
                tf.assign(slack_tpr, tf.clip_by_value(slack_tpr, 0.001, 0.999)))
            projection_ops.append(
                tf.assign(slack_tnr, tf.clip_by_value(slack_tnr, 0.001, 0.999)))
            
            # Set up 1 - G-mean objective.
            objective = tfco.wrap_rate(1.0 - tf.sqrt(slack_tpr * slack_tnr))

            # Set up slack constraints.
            constraints = []
            constraints.append(tfco.wrap_rate(slack_tpr) <= true_positive_rate)
            constraints.append(tfco.wrap_rate(slack_tnr) <= true_negative_rate)

            # Set up fairness equal-opportunity constraints.
            constraints.append(true_positive_rate0 <= true_positive_rate1 + epsilon)
            constraints.append(true_positive_rate1 <= true_positive_rate0 + epsilon)

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
                
                # Checkpoint once in 100 iterations.
                if ii % 100 == 0:
                    # Model weights.
                    model = [session.run(weights), session.run(threshold)]
                    models.append(model)

                    # Snapshot performace
                    error, tpr0, tpr1 = evaluate_expected_results(
                        train_set, [model], [1.0])
                    objectives.append(error)
                    violations.append([tpr0 - tpr1 - epsilon, tpr1 - tpr0 - epsilon])
                
            # Use the recorded objectives and constraints to find the best iterate.
            # Best model
            best_iterate = tfco.find_best_candidate_index(
                np.array(objectives), np.array(violations))
            best_model = models[best_iterate]
            
            # Stochastic model over a subset of classifiers.
            probabilities = tfco.find_best_candidate_distribution(
                np.array(objectives), np.array(violations))
            models_pruned = [models[i] for i in range(
                len(models)) if probabilities[i] > 0.0]
            probabilities_pruned = probabilities[probabilities > 0.0]

            # Stochastic model over all classifiers.
            probabilities_all = probabilities * 0.0 + 1.0 / len(probabilities)
                
            # Return Pruned models, Avg models, Best model
            results = {
                'stochastic': (models, probabilities_all),
                'pruned': (models_pruned, probabilities_pruned),
                'best': best_model,
                'objectives': objectives,
                'violations': violations
            }
            return results

        lr_range = [0.001, 0.01, 0.1]
        grid = [(xx, yy) for xx in lr_range for yy in lr_range]
        objectives = []
        violations = []

        for (lr, lr_con) in grid:
            print(lr, lr_con)
            results = lagrangian_optimizer(
                train_set, epsilon=epsilon, learning_rate=lr, 
                learning_rate_constraint=lr_con)
            error, tpr0, tpr1 = evaluate_expected_results(
                train_set, results['pruned'][0], results['pruned'][1])
            objectives.append(error)
            violations.append([tpr0 - tpr1 - epsilon, tpr1 - tpr0 - epsilon])

        best_index = tfco.find_best_candidate_index(
            np.array(objectives), np.array(violations), rank_objectives=True)
        results = lagrangian_optimizer(
            train_set, epsilon=epsilon, learning_rate=grid[best_index][0], 
            learning_rate_constraint=grid[best_index][1])
        print_results(
                    train_set, test_set, results['pruned'], results['objectives'], 
                    results['violations'])
    
    np.save("../new-results/3player_" + dataset_name + "_train_scores.npy", train_scores)
    np.save("../new-results/3player_" + dataset_name + "_test_scores.npy", test_scores)

    np.save("../new-results/3player_" + dataset_name + "_train_cons.npy", train_cons_val)
    np.save("../new-results/3player_" + dataset_name + "_test_cons.npy", test_cons_val)