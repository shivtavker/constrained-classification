# Commented out IPython magic to ensure Python compatibility.
import math
import random
import numpy as np
import pandas as pd
from six.moves import xrange
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco
import warnings

import time

tf.disable_eager_execution()

warnings.filterwarnings('ignore')
# %matplotlib inline

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

for d in ["covtype"]:
# for d in ["MACHO"]:
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()

    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    test_scores = []

    train_cons = []
    test_cons = []

    global_best_index = 0

    for global_i in [2]:
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=global_i)

        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        train_set = X_train, y_train
        test_set = X_test, y_test

        y_train.shape

        """### Evaluation functions

        """

        def error_rate(features, labels, model_weights, threshold=0):
            predictions = np.dot(features, model_weights) + threshold
            #   print(predictions.shape)
            y_pred = np.argmax(predictions, axis=1)
            return np.mean(y_pred != labels)

        def expected_error_rate(features, labels, models, probabilities):
            er = 0.0
            for i in range(len(models)):
                er += probabilities[i] * error_rate(
                    features, labels, models[i][0], models[i][1])
            return er


        def hmean_error(features, labels, models, probabilities):
            rates = [0]*num_class
            for i in range(num_class):
                rates[i] = 1 - expected_error_rate(
            features[labels == i, :],  labels[labels == i], models, probabilities)
        
            part_sum = 0
            for i in range(num_class):
                part_sum += 1/(rates[i] + 1e-7)
            
            return 1 - num_class*(1/part_sum)

        def expected_coverage(features, labels, models, probabilities):
            prs = [0]*num_class
            
            for i in range(num_class):
                prs[i] = 1 - expected_error_rate(features, [i]*len(labels), models,  probabilities)
        #     print(prs)
            return prs

        def coverage_value(posteriors, targets):
            return max(np.abs(posteriors - targets))


        def evaluate_expected_results(dataset, model, probabilities):
            x, y = dataset
            error = hmean_error(x, y, model, probabilities)
            coverage = expected_coverage(x, y, model, probabilities)
            return error, coverage

        def print_results(train_set, test_set, targets, epsilon, stochastic_model,
                        objectives=None, violations=None):
            x_train, y_train = train_set
            x_test, y_test = test_set

            models, probabilities = stochastic_model
            title = 'TFCO'
            
            # Plot objective:
            if objectives is not None:
                ff, ax = plt.subplots(1, 2, figsize=(8, 4))
                ax[0].plot(1 + np.arange(len(objectives)), objectives)
                ax[0].set_xlabel("Iteration Count")
                ax[0].set_title("1 - H-mean")
                ax[1].plot(1 + np.arange(len(violations)), np.max(violations, axis=1))
                ax[1].set_xlabel("Iteration Count")
                ax[1].set_title("Violations")
            
            error_train, coverage_train = evaluate_expected_results(
                train_set, models, probabilities)
            print(title + ": Train Hmean Score = %.3f" % (1-error_train))
            print(title + ": Train coverage value = %.3f" % coverage_value(coverage_train, targets))
            print()

            error_test, coverage_test = evaluate_expected_results(
                test_set, models, probabilities)
            print(title + ": Test Hmean Score = %.3f" % (1-error_test))
            print(title + ": Test coverage value = %.3f" % coverage_value(coverage_test, targets))

        """### 3-Player: Optimize H-mean s.t. Equal Opportunity Constraint"""

        def lagrangian_optimizer(train_set, num_class, targets, epsilon=0.01, 
                                learning_rate=0.01, learning_rate_constraint=0.01, 
                                loops=2000):
            tf.reset_default_graph()
            
            x_train, y_train = train_set
            num_examples = x_train.shape[0]
            dimension = x_train.shape[-1]
            
            # Data tensors.
            features_tensor = tf.constant(x_train.astype("float32"), name="features")
            # (One-hot) Labels: num_train x num_class
            labels_tensor = tf.one_hot(y_train, depth=num_class)

            # Linear model.
            weights = tf.Variable(tf.zeros([dimension, num_class], dtype=tf.float32), name="weights")
            threshold = tf.Variable(tf.zeros([num_class], dtype=tf.float32), name="threshold")
            predictions_tensor = (
                tf.matmul(features_tensor, weights) + threshold)
            
            # Class probabilities: num_train x num_class
            # probabilities_tensor = tf.nn.softmax(predictions_tensor)
            probabilities_tensor = predictions_tensor  # HN: changed
            
            # Set up rates.
            context = tfco.multiclass_rate_context(
                num_class, probabilities_tensor, labels_tensor)
            sc_loss = tfco.SoftmaxLoss()
            tprs = [tfco.true_positive_rate(
                context, i, penalty_loss=sc_loss) for i in range(num_class)]
            pprs = [tfco.positive_prediction_rate(
                context, i, penalty_loss=sc_loss) for i in range(num_class)]

            # Set up slack variables.
            slack_tprs = [tf.Variable(0.5, dtype=tf.float32) for i in range(num_class)]
            
            # Projection ops for slacks.
            projection_ops = [tf.assign(
                slack, tf.clip_by_value(slack, 0.001, 0.999)) for slack in slack_tprs]
            
            # Set up 1 - H-mean objective.
            objective = tfco.wrap_rate(
                1 - num_class*tf.math.reciprocal(tf.reduce_sum(tf.math.reciprocal(slack_tprs))))

            # Set up slack constraints.
            constraints = [tfco.wrap_rate(
                slack_tprs[i]) <= tprs[i] for i in range(num_class)]
                
            # Set up fairness equal-opportunity constraints.
            for i in range(num_class):
                constraints.append(pprs[i] <= targets[i] + epsilon)
                constraints.append(pprs[i] >= targets[i] - epsilon)

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
            for ii in range(loops):
            #     print(ii)
            #     start_time = time.time()
                # Gradient update.
                session.run(train_op)
                # Projection.
                session.run(projection_ops)
            #     end_time = time.time()
                
            #     print(end_time - start_time)
                
                # Checkpoint once in 100 iterations.
                if ii % 100 == 0:
                # Model weights.
                    model = [session.run(weights), session.run(threshold)]
                    models.append(model)

                    # Snapshot performace
                    error, coverage = evaluate_expected_results(
                        train_set, [model], [1.0])
                    objectives.append(error)
                    violations.append(
                            [max(coverage[i] - targets[i] - epsilon, 
                                targets[i] - coverage[i] - epsilon) 
                            for i in range(num_class)])

                if ii % 100 == 0:
                    print("Step %d | H-mean error = %3f | Coverage violation = %.3f" % (
                        ii, objectives[-1], max(violations[-1])))
                
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

        num_class = len(np.unique(y_train))

        p = np.zeros((num_class,))
        for i in range(num_class):
            p[i] = (y_train == i).mean()

        p

        """### Tune for best hyper-parameter
        """

        epsilon = 0.01
            
        # targets = np.array([500]*5 + [2000]*5)/(500*5 + 2000*5)
        targets = p

        lr_range = [0.01]
        grid = [(xx, yy) for xx in lr_range for yy in lr_range]
        objectives = []
        violations = []

        if global_i < 1:
            for (lr, lr_con) in grid:
                # print('Learning rate = %.3f | Constraint learning rate = %.3f' % (lr, lr_con))
                results = lagrangian_optimizer(
                    train_set, num_class, targets, epsilon=epsilon, learning_rate=lr, 
                    learning_rate_constraint=lr_con, loops=2000)
                hmean_err, coverage = evaluate_expected_results(
                    train_set, results['stochastic'][0], results['stochastic'][1])
                objectives.append(hmean_err)
                violations.append(
                    [max(coverage[i] - targets[i] - epsilon, 
                        targets[i] - coverage[i] - epsilon) for i in range(num_class)])
                # print()

            best_index = tfco.find_best_candidate_index(
                np.array(objectives), np.array(violations), rank_objectives = False)
            
            global_best_index = best_index

        best_index = global_best_index
        # best_index = 1

        print('Retrain with learning rate (%.3f, %.3f)\n' % grid[best_index])
        results = lagrangian_optimizer(
            train_set, num_class, targets, epsilon=epsilon, 
            learning_rate=grid[best_index][0], 
            learning_rate_constraint=grid[best_index][1], loops=2000)

        """### Evaluate stochastic model"""

        # print_results(
        #     train_set, test_set, targets, epsilon, results['pruned'], results['objectives'], 
        #     results['violations'])

        train_score = 1 - hmean_error(train_set[0], train_set[1], results['pruned'][0], results['pruned'][1])
        train_dist = expected_coverage(train_set[0], train_set[1], results['pruned'][0], results['pruned'][1])
        train_con = coverage_value(train_dist, targets)

        test_score = 1 - hmean_error(test_set[0], test_set[1], results['pruned'][0], results['pruned'][1])
        test_dist = expected_coverage(test_set[0], test_set[1], results['pruned'][0], results['pruned'][1])
        test_con = coverage_value(test_dist, targets)

        train_scores.append(train_score)
            # print(train_score)
        train_cons.append(train_con)
        test_scores.append(test_score)
        test_cons.append(test_con)

    mu_score_train = round(np.mean(train_scores), 3)
    mu_score_test = round(np.mean(test_scores), 3)
    std_score_train = round(np.std(train_scores), 3)
    std_score_test = round(np.std(test_scores), 3)

    mu_cons_train = round(np.mean(train_cons), 3)
    mu_cons_test = round(np.mean(test_cons), 3)
    std_cons_train = round(np.std(train_cons), 3)
    std_cons_test = round(np.std(test_cons), 3)

    print("===Hmean===")
    print(str(mu_score_train) + " (" + str(std_score_train) + ")")
    print(str(mu_score_test) + " (" + str(std_score_test) + ")")
    print("===Constraint===")
    print(str(mu_cons_train) + " (" + str(std_cons_train) + ")")
    print(str(mu_cons_test) + " (" + str(std_cons_test) + ")")

    np.save("3plib_" + d + "_train_scores.npy", train_scores)
    np.save("3plib_" + d + "_test_scores.npy", test_scores)

    np.save("3plib_" + d + "_train_cons.npy", train_cons)
    np.save("3plib_" + d + "_test_cons.npy", test_cons)
