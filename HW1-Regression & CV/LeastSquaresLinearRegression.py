'''
Test Case
---------
>>> prng = np.random.RandomState(0)
>>> N = 100

>>> true_w_F = np.asarray([1.1, -2.2, 3.3])
>>> true_b = 0.0
>>> x_NF = prng.randn(N, 3)
>>> y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

>>> linear_regr = LeastSquaresLinearRegressor()
>>> linear_regr.fit(x_NF, y_N)

>>> yhat_N = linear_regr.predict(x_NF)
>>> np.set_printoptions(precision=3, formatter={'float':lambda x: '% .3f' % x})
>>> print(linear_regr.w_F)
[ 1.099 -2.202  3.301]
>>> print(np.asarray([linear_regr.b]))
[-0.005]
'''

import numpy as np
# No other imports allowed!


class LeastSquaresLinearRegressor(object):
    ''' A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:

        .. math:
            \min_{w \in \mathbb{R}^F, b \in \mathbb{R}}
                \sum_{n=1}^N (y_n - b - \sum_f x_{nf} w_f)^2
        '''
        # Get the number of rows (N) and columns (F) from x_NF
        N, F = x_NF.shape

        # Add a column of ones to x_NF to account for bias
        x_with_bias = np.concatenate((x_NF, np.ones((N, 1))), axis=1).copy()

        # Compute the dot product of the transposed x_with_bias and itself
        x_t_x = np.dot(x_with_bias.T, x_with_bias).copy()

        # Compute the dot product of the transposed x_with_bias and y_N
        x_t_y = np.dot(x_with_bias.T, y_N).copy()

        # Solve for theta (model parameters) using the linear equation
        theta = np.linalg.solve(x_t_x, x_t_y).copy()

        # Extract the weights (w_F) and bias (b) from theta
        self.w_F = theta[:-1]
        self.b = theta[-1]

        # Print the weights and bias
        # print("Weights (w_F):", w_F)
        # print("Bias (b):", b)

    def predict(self, x_MF):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_MF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_M : 1D array, size M
            Each value is the predicted scalar for one example
        '''

        prediction = np.asarray([0.0])

        if self.w_F is not None and self.b is not None:

            # Apply coefficents to features, sum all in each row, and add bias
            prediction = np.sum(x_MF * self.w_F, axis=1) + self.b

            print(prediction)

        return prediction


def test_on_toy_data(N=100):
    '''
    Simple example use case
    With toy dataset with N=100 examples
    created via a known linear regression model plus small noise
    '''
    prng = np.random.RandomState(0)

    true_w_F = np.asarray([1.1, -2.2, 3.3])
    true_b = 0.0
    x_NF = prng.randn(N, 3)
    y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)

    yhat_N = linear_regr.predict(x_NF)

    np.set_printoptions(precision=3, formatter={
        'float': lambda x: '% .3f' % x})

    print("True weights")
    print(true_w_F)
    print("Estimated weights")
    print(linear_regr.w_F)

    print("True intercept")
    print(np.asarray([true_b]))
    print("Estimated intercept")
    print(np.asarray([linear_regr.b]))


if __name__ == '__main__':
    test_on_toy_data()
