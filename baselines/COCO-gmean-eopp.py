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

for dataset_name in ["compas"]:

  ### Data Loading 
  data_dict = np.load("../data/" + dataset_name + "_data.npy",  allow_pickle=True).item()
  x_train = data_dict.get('X_train')
  y_train = data_dict.get('y_train')
  z_train = x_train[:, 0].astype(int)
  train_set = x_train, y_train, z_train

  lmo_calls = 0


  x_test = data_dict['X_test']
  y_test = data_dict['y_test']
  z_test = x_test[:, 0].astype(int)
  test_set = x_test, y_test, z_test

  epsilon = 0.05
  T = 2000

  def evaluate_conf(y0, y1, z, y_prob, threshold0, threshold1):
    global lmo_calls
    lmo_calls += 1
    y_pred0 = 1.0 * (y_prob[z == 0] > threshold0)
    y_pred1 = 1.0 * (y_prob[z == 1] > threshold1)
    
    fp0 = np.mean((y0 == 0) & (y_pred0 == 1))
    fp1 = np.mean((y1 == 0) & (y_pred1 == 1))
    fn0 = np.mean((y0 == 1) & (y_pred0 == 0))
    fn1 = np.mean((y1 == 1) & (y_pred1 == 0))
    
    return fp0, fp1, fn0, fn1

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
    # print("Train gmean error = %.3f" % error_train)
    # print("Train fairness violation = %.3f (%.3f, %.3f)" % 
    #       (abs(tpr0_train - tpr1_train), tpr0_train, tpr1_train))
    # print()
    print("%.3f(%.3f)"% (error_train, abs(tpr0_train - tpr1_train)))

    error_test, tpr0_test, tpr1_test = evaluate_expected_results(
        test_set, models, probabilities)
    # print("Test gmean error = %.3f" % error_test)
    # print("Test fairness violation = %.3f (%.3f, %.3f)" % 
    #       (abs(tpr0_test - tpr1_test), tpr0_test, tpr1_test))

    # print("%.3f(%.3f)"% (error_test, abs(tpr0_test - tpr1_test)))

  def coco(train_set, test_set, model_unc, epsilon=0.05, 
                      learning_rate = 0.5, loops=10):
    delta_ = 1e-10
    # Skip iterations = (T)^1/3
    # How often should the Lagrange multiplier be updated.
    skip_iter = int(np.cbrt(T))

    # Datasets
    x_train, y_train, z_train = train_set
    # Vali same as training set.
    x_vali, y_vali, z_vali = train_set
    x_test, y_test, z_test = test_set

    # Append z to x as a column.
    x_train_ = np.concatenate([x_train, z_train.reshape(-1, 1)], axis=1)
    x_vali_ = np.concatenate([x_vali, z_vali.reshape(-1, 1)], axis=1)
    x_test_ = np.concatenate([x_test, z_test.reshape(-1, 1)], axis=1)
    train_set_ = x_train_, y_train, z_train
    vali_set_ = x_vali_, y_vali, z_vali
    test_set_ = x_test_, y_test, z_test
    
    # Labels for each group.
    y0 = y_train[z_train == 0]
    y1 = y_train[z_train == 1]
    
    # CPE model.
    weights, threshold = model_unc
    y_prob = np.dot(x_train, weights) + threshold
    
    # Label proportions.
    p = y_train.mean()
    p0 = np.mean(y0 == 1)
    p1 = np.mean(y1 == 1)
    
    # Group proportions.
    g0 = np.mean(z_train == 0)
    g1 = np.mean(z_train == 1)
    
    # Initialization.
    threshold0_temp = 0.5 
    threshold1_temp = 0.5 
    models = []
    objectives = []
    violations = []
    
    # Initialize conf matrix.
    fp0, fp1, fn0, fn1 = evaluate_conf(
        y0, y1, z_train, y_prob, threshold0_temp, threshold1_temp)

    #Initialize lagrange multipliers.
    lambda0 = 0.0
    lambda1 = 0.0

    inner_probabilities = np.zeros(loops)
    outer_probabilities = np.zeros(loops)

    for ii in range(1, loops+1):
      # G-mean gradient.
      tpr = (1 - g0 * fn0 / p - g1 * fn1 / p) + delta_
      tnr = (1 - g0 * fp0 / (1 - p) - g1 * fp1 / (1 - p)) + delta_
      coef_tpr = 0.5 * np.sqrt(tnr / tpr)
      coef_tnr = 0.5 * np.sqrt(tpr / tnr)

      # Minimize over confusion matrices.
      coef_fn0 = g0 * coef_tpr / p + (lambda0 / p0 - lambda1 / p0)
      coef_fn1 = g1 * coef_tpr / p + (lambda1 / p1 - lambda0 / p1)
      coef_fp0 = g0 * coef_tnr / (1 - p)
      coef_fp1 = g1 * coef_tnr / (1 - p)

      # Opt thresholds for cost-sensitive problem.
      if min(coef_fp0, coef_fn0) < 0:
        if coef_fp0 < coef_fn0:
          threshold0_temp = 1e-5
        else:
          threshold0_temp = 1 - 1e-5
      else:
        threshold0_temp = coef_fp0 / (coef_fp0 + coef_fn0 + delta_)
        threshold0_temp = min(threshold0_temp, 1 - delta_)
        threshold0_temp = max(threshold0_temp, delta_)
      threshold0 = np.log(threshold0_temp / (1 - threshold0_temp)) 

      if min(coef_fp1, coef_fn1) < 0:
        if coef_fp1 < coef_fn1:
          threshold1_temp = 1e-5
        else:
          threshold1_temp = 1 - 1e-5
      else:
        threshold1_temp = coef_fp1 / (coef_fp1 + coef_fn1 + delta_)
        threshold1_temp = min(threshold1_temp, 1 - delta_)
        threshold1_temp = max(threshold1_temp, delta_)
      threshold1 = np.log(threshold1_temp / (1 - threshold1_temp))

      # Evaluate metrics.
      fp0_hat, fp1_hat, fn0_hat, fn1_hat = evaluate_conf(
          y0, y1, z_train, y_prob, threshold0, threshold1)

      fp0 = (1 - 2.0 / (ii + 1)) * fp0 + 2.0 / (ii + 1) * fp0_hat
      fp1 = (1 - 2.0 / (ii + 1)) * fp1 + 2.0 / (ii + 1) * fp1_hat
      fn0 = (1 - 2.0 / (ii + 1)) * fn0 + 2.0 / (ii + 1) * fn0_hat
      fn1 = (1 - 2.0 / (ii + 1)) * fn1 + 2.0 / (ii + 1) * fn1_hat

      inner_probabilities[:ii-1] *= (1 - 2.0 / (ii + 1))
      inner_probabilities[ii - 1] = 2.0 / (ii + 1)

      # Thresholds are added not subtracted during evaluation.
      weights_ = np.concatenate([weights, [-threshold1 + threshold0]])
      threshold_ = threshold - threshold0
      model = [weights_, threshold_]
      models.append(model)

      # Evaluate metrics.
      error, tpr0, tpr1 = evaluate_expected_results(
          train_set_, [model], [1.0])
      objectives.append(error)
      violations.append([tpr0 - tpr1 - epsilon, tpr1 - tpr0 - epsilon])

      # # Report once in 25 iterations.
      # if ii % 25 == 0:
      #   print("Step %d | G-mean error = %3f | EO violation = %.3f" % (
      #         ii, objectives[-1], max(violations[-1])))

      if ii % skip_iter == 0:
        # Update lambda.
        lambda0 += learning_rate * (fn0 / p0 - fn1 / p1 - epsilon)
        lambda1 += learning_rate * (fn1 / p1 - fn0 / p0 - epsilon)

        # Project lambdas.
        lambda0 = np.maximum(lambda0, 0.0)
        lambda1 = np.maximum(lambda1, 0.0)

        # Update count.
        outer_probabilities += inner_probabilities 

    # Normalize probabilities to sum to 1.
    if ii % skip_iter != 0:  # Last outer iteration did not complete.
      outer_probabilities += inner_probabilities 
    outer_probabilities *= 1.0 / np.sum(outer_probabilities)

    probabilities_pruned = tfco.find_best_candidate_distribution(
        np.array(objectives), np.array(violations))
    
    # Shrinking.
    models_pruned = [models[jj] for jj in range(len(models)) if probabilities_pruned[jj] > 0]
    probabilities_pruned = probabilities_pruned[probabilities_pruned > 0]
    
    # Best model.
    best_index = tfco.find_best_candidate_index(
        np.array(objectives), np.array(violations))

    # Return Pruned models, Avg models, Best model
    results = {
        'stochastic': (models, outer_probabilities),
        'pruned': (models_pruned, probabilities_pruned),
        'best': models[best_index],
        'objectives': objectives,
        'violations': violations,
        'modified_train_set': train_set_,
        'modified_test_set': test_set_
    }
    return results

    # Unconstrained LogReg
  logreg_model = linear_model.LogisticRegressionCV()
  logreg_model.fit(x_train, y_train)
  model_unc = (logreg_model.coef_.reshape(-1,), logreg_model.intercept_)
  results = coco(train_set, test_set, model_unc)
  # print("LMO Calls: ", lmo_calls)
  # np.save("./results/coco-compas-2.npy", [results['objectives'], results['violations']])

  lr_range = [1e-3, 0.01, 0.1, 0.5, 1.0, 10, 20]
  # grid = [xx for xx in lr_range]

  for T in [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 2000]:
    objectives = []
    violations = []
    for lr in lr_range:
        # print(lr, lr_con)
        results = coco(
            train_set, test_set, model_unc, epsilon=epsilon, learning_rate=lr, loops=T)
        error, tpr0, tpr1 = evaluate_expected_results(
            results['modified_train_set'], results['stochastic'][0], results['stochastic'][1])
        objectives.append(error)
        violations.append([tpr0 - tpr1 - epsilon, tpr1 - tpr0 - epsilon])

    best_index = tfco.find_best_candidate_index(
        np.array(objectives), np.array(violations), rank_objectives=False)
    # print(best_index)
    # print("Learning Rate Used: ", lr_range[best_index])
    results = coco(
            train_set, test_set, model_unc, epsilon=epsilon, learning_rate=lr_range[best_index], loops=T)

    print_results(
        results['modified_train_set'], results['modified_test_set'], 
        results['stochastic'], results['objectives'], results['violations'])