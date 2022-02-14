---
title: "Linear Regression Part 8 - Stochastic Gradient Descent"
date: 2022-01-31T16:24:09+01:00
tags: [machine-learning, linear-regression, python]
draft: false
---

In a previous post, we have discussed the gradient descent algorithm for linear regression applied to multiple features. As the size of the training dataset increases, gradient descent becomes very computationally expensive. In particular, if we consider the computation of the cost function,

$$J = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}^{(i)} - {y}^{(i)} \right)^2$$,

which is a calculation that we need to compute for each training data iteration; we can can see that the cost function is expensive as m increases.

Stochastic gradient descent is a variant of gradient descent where **coefficients are updated after each observation**, and it is therefore better suited for large datasets. Therefore if we consider an observation $i$ of the training dataset, we can calculate the cost of the observation, or how well our hypothesis function predicts the observation as,

$$Cost(\beta, (x^{(i)}, y^{(i)})) = \frac{1}{2} \left(\hat{y}^{(i)} - {y}^{(i)} \right)^2$$

We can therefore calculate the cost function as

$$J(\beta) = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - {y}^{(i)} \right)^2$$

Random Gradient Descent can be implemented as follows:

- Shuffle the training dataset, thus making sure that the order of the observations is random.

- For each observation:

  - $$\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix} :=
\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix} -
\alpha
\begin{pmatrix}
x_0^{(i)}\\\\
x_1^{(i)}\\\\
\vdots\\\\
x_n^{(i)}
\end{pmatrix}
\cdot
\begin{pmatrix}
\hat{y}^{(i)} - {y}^{(i)}
\end{pmatrix}
$$

Where $m$ is the number of training data points.

We can execute the above equation for a fixed number of iterations, where convergence is usually achieved. The code to implement this algorithm is as follows:

```Python

import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np


def stochastic_gradient_descent(file, alpha, max_epochs = 5):
    '''
    Implementation of Stochastic Gradient Descent
    
    Exit condition: max_epochs epochs
    '''
    
    np.random.seed(42)

    # load the training data
    full_filename = os.path.join(os.path.dirname(__file__), file)
    data_set = pd.read_csv(full_filename, delimiter=',', header=0, index_col=False)

    # training_data = training_data.sample(frac=1).reset_index(drop=True)

    # divide the data into features and labels
    X = data_set.drop(['y'], axis=1).to_numpy()
    
    # add a column of ones to the features matrix to account for the intercept, a0
    X = np.insert(X, 0, 1, axis=1)

    Y = data_set['y'].to_numpy()
    
    # length of the training data
    m = len(Y)

    # initialize the y_hat vector to 0
    y_hat = np.zeros(len(Y))
    
    # beta will hold the values of the coefficients, hence it will be  the size 
    # of a row of the X matrix
    # initialize beta to random values
    beta = np.random.random(len(X[0]))

    # initialize the number of epochs
    epochs = 0

    # loop until exit condition is met
    while True:
        
        i = np.random.randint(0, m)

        # print(f'Minibatch: {i}')
        x = X[i]
        y = Y[i]

        # calculate the hypothesis function for all training data
        y_hat = np.dot(beta, x.T)

        #  calculate the residuals
        residuals = y_hat - y

        # calculate the new value of beta
        beta -= (alpha * residuals * x)

        epochs += 1
  
        # check if the cost function is close enough to 0, if so, break or if the number of 
        # iterations is greater than the threshold, break
        if epochs > (m*max_epochs):
            break
    
    # calculate the cost for the training data and return the beta values and 
    # the number of iterations and the cost
    y_hat = np.dot(beta, X.T)
    residuals = y_hat - Y
    cost = np.dot(residuals, residuals) / ( 2 * m)
    
    return beta, cost


if __name__ == '__main__':

    from timeit import default_timer as timer

    file = 'data.csv'
    alpha = 0.0001
    max_epochs = 5
    start = timer()
    beta, cost = stochastic_gradient_descent(file, alpha, max_epochs) 
    end = timer()
    print(f'Time: {end - start}, beta: {beta}, cost: {cost}')
```

### Exit Criteria Considerations

A critical consideration in the above algorithm is that the aspect of convergence measurement is wholly taken out, and the algorithm terminates after a fixed number of epochs. A better approach is to reserve a portion of the data for validation and then use the remaining data for training. The validation set is then used to calculate the cost function and the algorithm terminates when the cost function converges. An implementation of this approach is as follows:

```Python

import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pandas as pd
import sys
import numpy as np


def stochastic_gradient_descent(file, alpha=0.0023, epochs_threshold=100, costdifference_threshold=0.00001, plot=False):
    '''
    Implementation of Stochastic Gradient Descent
    
    '''

    np.random.seed(42)
    
    # load the training data
    full_filename = os.path.join(os.path.dirname(__file__), file)
    data_set = pd.read_csv(full_filename, delimiter=',', header=0, index_col=False)

    # create train and test sets
    mask = np.random.rand(len(data_set)) < 0.8
    training_data = data_set[mask]
    validation_data = data_set[~mask]

    # divide the data into features and labels
    X_train = data_set.drop(['y'], axis=1).to_numpy()
    # add a column of ones to the features matrix to account for the intercept, a0
    X_train = np.insert(X_train, 0, 1, axis=1)
    Y_train = data_set['y'].to_numpy()

    X_validation = validation_data.drop(['y'], axis=1).to_numpy()
    X_validation = np.insert(X_validation, 0, 1, axis=1)
    Y_validation = validation_data['y'].to_numpy()


    # length of the training data
    m = len(Y_train)

    # initialize the y_hat vector to 0
    y_hat = np.zeros(len(Y_train))
    
    # beta will hold the values of the coefficients, hence it will be  the size 
    # of a row of the X matrix
    # initialize beta to random values
    beta = np.random.random(len(X_train[0]))

    # initialize the number of epochs
    count = 0

    previous_validation_cost = sys.float_info.max

    # loop until exit condition is met
    while True:
        
        i = np.random.randint(0, m)

        # print(f'Minibatch: {i}')
        x = X_train[i]
        y = Y_train[i]

        # calculate the hypothesis function for all training data
        y_hat = np.dot(beta, x.T)

        #  calculate the residuals
        residuals = y_hat - y

        # calculate the new value of beta
        beta -= (alpha * residuals * x)

        count += 1

        if count % 1000 == 0:
            y_hat_validation = np.dot(beta, X_validation.T)
            residuals_validation = y_hat_validation - Y_validation
            cost_validation = np.dot(residuals_validation, residuals_validation) / ( 2 * len(Y_validation))
            
            if abs(previous_validation_cost - cost_validation) < costdifference_threshold:
                break
            else:
                previous_validation_cost = cost_validation
            
            # uncomment this line to see details
            # print(f'Epoch: {count/m} Cost: {cost_validation} beta: {beta}')
                    
        # check if the cost function is close enough to 0, if so, break or if the number of 
        # iterations is greater than the threshold, break
        if (count/m) > (epochs_threshold):
            break
    
    # calculate the cost for the training data and return the beta values and 
    # the number of iterations and the cost
    y_hat = np.dot(beta, X_train.T)
    residuals = y_hat - Y_train
    cost = np.dot(residuals, residuals) / ( 2 * m)
    
    return beta, count, cost


if __name__ == '__main__':

    from timeit import default_timer as timer

    file = 'data.csv'
    alpha = 0.00033
    epochs_threshold = 100
    costdifference_threshold = 0.005
    plot = False


    start = timer()
    beta, count, cost = stochastic_gradient_descent(file, alpha, epochs_threshold, costdifference_threshold, plot)
    end = timer()
    print(f'Time: {end - start}, beta: {beta}, count: {count}, cost: {cost}')
```

### Conclusion

In this post, we have seen how the coefficients of the linear regression model can be optimized using the stochastic gradient descent algorithm. We have also seen different ways to determine exit criteria for the algorithm. We have also seen how to implement the algorithm in Python.
