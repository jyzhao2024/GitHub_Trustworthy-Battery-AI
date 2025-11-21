import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        # Initialize weights and bias as None
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model: calculate weights and bias using least squares method
        :param X: Feature data (n_samples, n_features)
        :param y: Target values (n_samples,)
        """
        # Add a column of ones to X for bias calculation
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        # Calculate optimal weight parameters using Normal Equation
        # Formula: θ = (X^T * X)^(-1) * X^T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        # Separate weights and bias
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """
        Make predictions using the trained model
        :param X: Feature data (n_samples, n_features)
        :return: Predicted values (n_samples,)
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b.dot(np.r_[self.bias, self.weights])  # Calculate predicted values