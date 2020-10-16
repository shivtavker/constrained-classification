# Constrained-Classification

This repository contains code for SBFW Algorithm presented in Consistent Plugin Classifier for Complex Objectives and Constraints.

To replicate the experiments presented in the paper run

```python
python run_experiments.py
```

We cover the following loss functions:

- H-mean loss ("hmean")
- Q-mean loss ("qmean")
- G-mean Loss ("gmean")
- Linear 0-1 Loss ("linear") -- For this you may want to provide a alternate loss implying the complex loss value when linear loss is minimized.

and following constraints:

- Demographic Parity ("DP")
- Coverage ("COV")
- Equal Oppotunity ("EOpp")
- KL-divergence ("KLD")

We provide a function _SBFW_ that can be imported from algorithm.SBFW. It takes in the following Inputs

- X_train, y_train
- X_test, y_test
- Loss Name
- Constraint Name
- Lambda (Hyperparameter - Default value works fine)
- Epsilon (Slack depends on how precisely do we want the constraints to be satisfied - Lower values take longer iterations)
- Eta_t_arr (Can be a length 1 array)
- Total iterations (T)

```python
loss_train, loss_test, constraint_value_train, constraint_value_test = SBFW(X_train, y_train, X_test, y_test, "gmean", "EOpp", lambda_val, epsilon, eta_t_array, T)
```
