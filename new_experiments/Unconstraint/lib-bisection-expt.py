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
    data_dict = np.load("./data/"+ d +"_data.npy", allow_pickle=True).item()

    X_data = data_dict['X']
    Y_data = data_dict['Y']

    train_scores = []
    test_scores = []

    for i in range(1):
        u, count = np.unique(Y_data, return_counts=True)
        count_sort_ind = np.argsort(-count)
        classes = list(u[count_sort_ind])

        def func_map(x):
            return classes.index(x)

        Y_data = np.vectorize(func_map)(Y_data)

        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)
        
        if d == "MACHO":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

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

        def get_confusion_matrix(features, labels, model_weights, threshold=0):
            predictions = np.dot(features, model_weights) + threshold
            y_pred = np.argmax(predictions, axis=1)
            confusion_matrix = np.zeros(shape=(num_class, num_class))
            np.add.at(confusion_matrix, (labels.astype(int), y_pred.astype(int)), 1)

            return confusion_matrix/len(labels)

        def expected_error_rate(features, labels, models, probabilities):
            er = 0.0
            for i in range(len(models)):
                er += probabilities[i] * error_rate(
                    features, labels, models[i][0], models[i][1])
            return er

        def fmeasure_error(features, labels, models, probabilities):
            """Returns expected F-measure for stochastic model."""

            C = np.zeros(shape=(num_class, num_class))
            for i in range(len(models)):
                C += probabilities[i] * get_confusion_matrix(
                    features, 
                    labels, 
                    models[i][0], 
                    models[i][1]
                )
            
            num_part = 0.0
            for i in range(1, num_class):
                num_part += 2*C[i ,i]
            
            denom_part = 2.0
            for i in range(num_class):
                denom_part -= C[0, i] + C[i, 0]
            
            return 1 - num_part/(denom_part+1e-7)

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
            error = fmeasure_error(x, y, model, probabilities)
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
            print(title + ": Train F-Score = %.3f" % (1-error_train))
            print(title + ": Train coverage value = %.3f" % coverage_value(coverage_train, targets))
            print()

            error_test, coverage_test = evaluate_expected_results(
                test_set, models, probabilities)
            print(title + ": Test F-Score = %.3f" % (1-error_test))
            print(title + ": Test coverage value = %.3f" % coverage_value(coverage_test, targets))

        """### 3-Player: Optimize H-mean s.t. Equal Opportunity Constraint"""

        def lagrangian_optimizer(train_set, num_class, targets, epsilon=0.01, 
                         learning_rate=0.01, learning_rate_constraint=0.01, 
                         negative_class=0, loops=2000):
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

            # Set up rates.
            context = tfco.multiclass_rate_context(
                num_class, predictions_tensor, labels_tensor)
            sc_loss = tfco.SoftmaxCrossEntropyLoss()

            # Slacks.
            slack1 = tf.Variable(0.5)
            slack2 = tf.Variable(0.5)
            
            # Set up 1 - F-measure objective. 
            objective = tfco.wrap_rate(1.0 - slack1 / (0.00001 + slack2))
            Ciis =[tfco.true_positive_proportion(
                context, positive_class=ii) for ii in range(num_class) if ii != negative_class]
            C11 = tfco.true_positive_proportion(context, positive_class=negative_class)
            C1i_sum = C11 + tfco.false_negative_proportion(
                context, positive_class=negative_class)
            Ci1_sum = C11 + tfco.false_positive_proportion(
                context, positive_class=negative_class)
            constraints = []
            constraints.append(tfco.wrap_rate(slack1) <= 2 * sum(Ciis))
            constraints.append(tfco.wrap_rate(slack2) >= 2 - C1i_sum - Ci1_sum)
            
            # Projection ops for slacks.
            projection_ops = []
            projection_ops.append(
                tf.assign(slack1, tf.clip_by_value(slack1, 0.001, 0.999)))
            projection_ops.append(
                tf.assign(slack2, tf.clip_by_value(slack2, 0.001, 0.999)))

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
                    error, coverage = evaluate_expected_results(
                        train_set, [model], [1.0])
                    objectives.append(error)
                    violations.append([0.0])

                # if ii % 100 == 0:
                #     viols = session.run(problem.constraints())
                    # print("Step %d | F-measure error = %3f | Violation = %.3f" % (
                    #     ii, objectives[-1], np.max(viols)))
                    # s1, s2 = session.run([slack1, slack2])
                    # cc = get_confusion_matrix(x_train, y_train, model[0], model[1])
                    # print(cc[0, 0], np.sum(cc[0, 1:]))
                    # print(np.sum(cc[1:, 0]), np.sum(cc[1:, 1:]))
                    # num = (s1 - viols[0]) / 2
                    # den = viols[1] + s2
                    # print(1 - num / den)

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

        epsilon = 0.5
            
        # targets = np.array([500]*5 + [5000]*5)/(500*5 + 5000*5)
        targets = p

        lr_range = [0.001, 0.01, 0.1]
        grid = [(xx, yy) for xx in lr_range for yy in lr_range]
        objectives = []
        violations = []

        for (lr, lr_con) in grid:
            # print('Learning rate = %.3f | Constraint learning rate = %.3f' % (lr, lr_con))
            results = lagrangian_optimizer(
                train_set, num_class, targets, epsilon=epsilon, learning_rate=lr, 
                learning_rate_constraint=lr_con, loops=5000)
            hmean_err, coverage = evaluate_expected_results(
                train_set, results['stochastic'][0], results['stochastic'][1])
            objectives.append(hmean_err)
            violations.append(
                [max(coverage[i] - targets[i] - epsilon, 
                    targets[i] - coverage[i] - epsilon) for i in range(num_class)])
            # print()

        best_index = tfco.find_best_candidate_index(
            np.array(objectives), np.array(violations), rank_objectives = False)
        print('Retrain with learning rate (%.3f, %.3f)\n' % grid[best_index])

        # best_index = 0

        results = lagrangian_optimizer(
            train_set, num_class, targets, epsilon=epsilon, 
            learning_rate=grid[best_index][0], 
            learning_rate_constraint=grid[best_index][1], loops=5000)

        """### Evaluate stochastic model"""

        # print_results(
        #     train_set, test_set, targets, epsilon, results['pruned'], results['objectives'], 
        #     results['violations'])

        train_score = 1 - fmeasure_error(train_set[0], train_set[1], results['pruned'][0], results['pruned'][1])
        test_score = 1 - fmeasure_error(test_set[0], test_set[1], results['pruned'][0], results['pruned'][1])

        train_scores.append(train_score)
            # print(train_score)
        test_scores.append(test_score)

    mu_train = 1-round(np.mean(train_scores), 3)
    mu_test = 1-round(np.mean(test_scores), 3)
    std_train = round(2*np.std(train_scores)/np.sqrt(len(train_scores)), 3)
    std_test = round(2*np.std(test_scores)/np.sqrt(len(test_scores)), 3)

    print(str(mu_train) + " (" + str(std_train) + ")")
    print(str(mu_test) + " (" + str(std_test) + ")")
