class MajorityClassifier:
    def init(self):
        self.majority_class = None

    def fit(self, X, y):
        """
        Train the classifier by finding the majority class in the target labels.
        
        :param X: Feature matrix (ignored in majority classifier)
        :param y: List of class labels
        """
        values = set(y)
        self.majority_class = max(values, key = y.count)

    def predict(self, X):
        """
        Predict the majority class for all input samples.
        
        :param X: Feature matrix (ignored in majority classifier)
        :return: List with majority class predictions
        """
        if self.majority_class is None:
            raise ValueError("Classifier has not been trained yet.")
        return [self.majority_class] * len(X)