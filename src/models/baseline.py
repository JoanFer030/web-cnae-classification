from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        """
        Train the classifier by finding the majority class in the target labels.
        
        :param X: Feature matrix (ignored in majority classifier)
        :param y: List or array of class labels
        """
        y = list(y)
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        """
        Predict the majority class for all input samples.
        
        :param X: Feature matrix (ignored in majority classifier)
        :return: List with majority class predictions
        """
        if self.majority_class is None:
            raise ValueError("Classifier has not been trained yet.")
        return [self.majority_class] * len(X)