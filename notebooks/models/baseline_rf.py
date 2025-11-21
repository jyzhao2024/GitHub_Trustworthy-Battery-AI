import numpy as np
from sklearn.ensemble import RandomForestRegressor


class SimpleRandomForest:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize Random Forest regression model
        :param n_estimators: Number of trees in the forest
        :param max_depth: Maximum depth of each tree
        :param random_state: Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        """
        Train the model
        :param X: Feature data (n_samples, n_features)
        :param y: Target values (n_samples,)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model
        :param X: Feature data (n_samples, n_features)
        :return: Predicted values (n_samples,)
        """
        return self.model.predict(X)