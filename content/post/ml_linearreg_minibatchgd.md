---
title: "Linear Regression Part 9 - Mini Batch Gradient Descent"
date: 2022-02-14
tags: [machine-learning, linear-regression, gradient-descent, python]
draft: false
---

In the [last post](/post/ml_linearreg_stochasticgd.md) we compared the stochastic gradient descent algorithm to the batch gradient descent algorithm that we has discussed in [a previous post](/post/ml_linearreg_gradientdescent.md). We discussed that as the size of the training dataset increases, batch gradient descent, where we use all the examples of the training set in each iteration, becomes very computationally expensive and that we can therefore use stochastic gradient descent, where we use one example of the training set in each iteration, to have a more efficient way to approach the coefficients of our hypothesis function.

In this post we will discuss mini-batch gradient descent, where we use a number $k$ of the training set examples in each iteration, which is a variation of the thoughts of stochastic gradient descent. We will discuss the general idea of mini-batch gradient descent and how to implement it in Python.

Therefore we can consider our training dataset as a collection of $m/k$ mini batches;

$$\textbf{X} = \begin{pmatrix}
x_0^{(1)} &\dots & x_0^{(k)} & x_0^{(k+1)} & \dots & x_0^{(2k)} & \dots \dots & x_0^{(m)}\\\\
x_1^{(1)} &\dots & x_1^{(k)} & x_1^{(k+1)} & \dots & x_1^{(2k)} & \dots \dots & x_1^{(m)}\\\\
\vdots     & & \vdots     & \vdots       &  & \vdots      &  & \vdots    \\\\
x_n^{(1)} &\dots & x_n^{(k)} & x_n^{(k+1)} & \dots & x_n^{(2k)} & \dots \dots & x_n^{(m)}\\\\
\end{pmatrix}$$
$$Dim:[n \times m]$$

Therefore the matrix $X$ can be represented by a matrix of mini batches,

$$\textbf{X} = \begin{pmatrix}
X^{ \\{  1 \\} } & X^{ \\{  2 \\} } & \dots & X^{ \\{  m/k \\} }\\\\
\end{pmatrix}$$

where $X^{ \\{  1 \\} }$ represents the first mini batch, $X^{ \\{  2 \\} }$ represents the second mini batch, and so on, such that;

$$X^{ \\{  1 \\} } = \begin{pmatrix}
x_0^{(1)} &\dots & x_0^{(k)} \\\\
x_1^{(1)} &\dots & x_1^{(k)} \\\\
\vdots     & & \vdots        \\\\
x_n^{(1)} &\dots & x_n^{(k)} \\\\
\end{pmatrix}$$
$$Dim:[n \times k]$$

Similarly for the $Y$ vector,

$$\textbf{Y} = \begin{pmatrix}
Y^{ \\{  1 \\} } & Y^{ \\{  2 \\} } & \dots & Y^{ \\{  m/k \\} }\\\\
\end{pmatrix}$$

Calculating the hypothesis function for a mini-batch of training data, we can write the following equation:

We can now calculate the hypothesis function for the first mini batch as a matrix multiplication:

$$\begin{pmatrix}
\hat{y}^{(1)} \\\\
\hat{y}^{(2)} \\\\
\vdots \\\\
\hat{y}^{(k)}
\end{pmatrix} =
\begin{pmatrix}
x_0^{(1)} & x_1^{(1)} & \dots & x_n^{(1)} \\\\
x_0^{(2)} & x_1^{(2)} & \dots & x_n^{(2)} \\\\
\vdots & \vdots & \vdots & \vdots \\\\
x_0^{(k)} & x_1^{(k)} & \dots & x_n^{(k)}
\end{pmatrix}
\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix}
$$
$$Dim:[k \times n] \cdot [n \times 1]$$

$$ \hat{Y}^{ \\{ 1 \\}}  = \left( {X}^{ \\{ 1 \\}} \right)^T \beta$$

We can therefore update the coefficients of our hypothesis function from this mini-batch as follows:

$$\begin{pmatrix}
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
\frac{\alpha}{k}
\begin{pmatrix}
x_0^{(1)} & \dots & x_0^{(k)}\\\\
x_1^{(1)} & \dots & x_1^{(k)}\\\\
\vdots&&\vdots\\\\
x_n^{(1)} & \dots & x_n^{(k)}\\\\
\end{pmatrix}
\cdot
\begin{pmatrix}
\hat{y}^{(1)} - {y}^{(1)} \\\\
\hat{y}^{(2)} - {y}^{(2)} \\\\
\vdots \\\\
\hat{y}^{(k)} - {y}^{(k)}
\end{pmatrix}
$$
$$Dim:[n \times k] \cdot [k \times 1]$$

or

$$\beta := \beta -\frac{\alpha}{k} \textbf{X} \cdot \textbf{R}$$

The cost function for the mini-batch gradient descent algorithm is:

$$\textbf{J} = \frac{1}{k} \sum_{i=1}^{k} (\hat{y}^{(i)} - y^{(i)})^2 $$

We notice that this is very similar to batch gradient descent, but in this case we are considering only a batch of $k$ examples of the training set. In this case, however an epoch will update the coefficients of our hypothesis function $m/k$ times instead of only once as in the case of batch gradient descent.

The following code will implement mini-batch gradient descent.

```python
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

def minibatch_gradient_descent(file:str, alpha:float=0.0023, batch_size:int=100, epochs_threshold:int=100000, costdifference_threshold:float=0.00001, plot:bool=False):

    # load the training data
    full_filename = os.path.join(os.path.dirname(__file__), file)
    training_data = pd.read_csv(full_filename, delimiter=',', header=0, index_col=False)

    # training_data = training_data.sample(frac=1).reset_index(drop=True)

    # divide the data into features and labels
    X = training_data.drop(['y'], axis=1).to_numpy()
    # add a column of ones to the features matrix to account for the intercept, a0
    X = np.insert(X, 0, 1, axis=1)

    Y = training_data['y'].to_numpy()
    
    # length of the training data
    m = len(Y)
    print(f'Length of the training data: {m}')

    # initialize the y_hat vector to 0
    y_hat = np.zeros(len(Y))
    
    # beta will hold the values of the coefficients, hence it will be  the size 
    # of a row of the X matrix
    # initialize beta to random values
    beta = np.random.random(len(X[0]))

    # minibatches setting
    # number of minibatches = m => stochastic gradient descent
    # number of minibatches = 1 => batch gradient descent
    minibatch_size = int(m/batch_size)

    # initialize the number of epochs
    epoch_count = 0

    # initialize the previous cost function value to a large number
    # previous_cost = sys.float_info.max
    
    # store the cost function and a2 values for plotting
    costs = []
    a_2s = []
    
    previous_cumulative_cost = sys.float_info.max
    
    # loop until exit condition is met
    while True:

        cumulative_cost = 0

        for i in range(batch_size):

            # print(f'Minibatch: {i}')
            minibatch_X = X[i*minibatch_size:(i+1)*minibatch_size]
            minibatch_Y = Y[i*minibatch_size:(i+1)*minibatch_size]

            # calculate the hypothesis function for all training data
            y_hat = np.dot(beta, minibatch_X.T)
            #  calculate the residuals
            residuals = y_hat - minibatch_Y
            
            
            # calculate the new value of beta
            beta -= ( alpha / minibatch_size)  * np.dot(residuals, minibatch_X)

            # calculate the cost function
            cost = np.dot(residuals, residuals) / ( 2 * minibatch_size)

            cumulative_cost += cost

        # increase the number of iterations
        epoch_count += 1

        # record the cost and a1 values for plotting
        #     costs.append(cost)
        #     a_2s.append(__beta[2])

        cost_difference = previous_cumulative_cost - cumulative_cost
        # print(f'Epoch: {epochs}, average cost: {(cumulative_cost/minibatches_number):.3f}, beta: {beta}')
        previous_cumulative_cost = cumulative_cost

        # check if the cost function is diverging, if so, break
        # if cost_difference < 0:
        #     print(f'Cost function is diverging. Stopping training.')
        #     break
            
        # check if the cost function is close enough to 0, if so, break or if the number of 
        # iterations is greater than the threshold, break
        if abs(cost_difference) < costdifference_threshold or epoch_count > epochs_threshold:
            break

    # # plot the cost function and a1 values
    # plt.plot(a_2s[3:], costs[3:], '--bx', color='lightblue', mec='red')
    # plt.xlabel('a2')
    # plt.ylabel('cost')
    # plt.title(r'Cost Function vs. a1, with $\alpha$ =' + str(__alpha))
    # plt.show()
    
    # calculate the cost for the training data and return the beta values and 
    # the number of iterations and the cost
    y_hat = np.dot(beta, X.T)
    residuals = y_hat - Y
    cost = np.dot(residuals, residuals) / ( 2 * m)
    
    return beta, epoch_count, cost
    

if __name__ == '__main__':

    from timeit import default_timer as timer

    file = 'data.csv'
    alpha = 0.00023
    epochs_threshold = 1000
    costdifference_threshold = 0.00001
    plot = False
    batch_size = 100


    start = timer()
    beta, epoch_count, cost = minibatch_gradient_descent(file, alpha, batch_size, epochs_threshold, costdifference_threshold, plot)
    end = timer()
    print(f'Time: {end - start} beta: {beta}, epoch_count: {epoch_count}, cost: {cost}')
```
