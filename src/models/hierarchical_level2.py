from .baseline import MajorityClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, resampler=None):
        """
        Initialize the hierarchical classifier.

        :param base_model: A scikit-learn compatible classifier (must implement fit and predict)
        :param resampler: A resampling strategy (e.g., SMOTE) that implements fit_resample
        """
        if base_model is None:
            raise ValueError("Base model must be provided")
        if resampler is None:
            raise ValueError("Resampler must be provided")
        
        self.base_model = base_model
        self.resampler = resampler
        self.model_level_1 = None
        self.models_level_2 = {}

    def fit(self, train_level_2):
        """
        Fit the hierarchical model on two levels of training data.

        :param train_level_2: Dict {label: (X2, y2)} with level 2 training per level-1 label
        :return: self
        """
        # Initialize models for level 2 and store synthetic data generated
        self.models_level_2 = {}
        synthetic_data_level_2 = {"X": [], "y": []}
        # STEP 1: Train level 2 models
        for label, (X2, y2) in train_level_2.items():
            # Train level 2 model (MajorityClassifier)
            if len(np.unique(y2)) < 2:
                # If there"s only one class in the resampled data, use MajorityClassifier
                X2_res, y2_res = X2, y2
                model = MajorityClassifier()
                model.fit(X2_res, y2_res)
            else:
                X2_res, y2_res = self.resampler.fit_resample(X2, y2)
                # Train a base model for this level 2 label
                model = clone(self.base_model)
                model.fit(X2_res, y2_res)
            self.models_level_2[label] = model
            synthetic_data_level_2["X"].extend(X2_res)
            synthetic_data_level_2["y"].extend([label] * len(X2_res))
        # STEP 2: Train level 1 model using synthetic data generated from level 2
        X1 = synthetic_data_level_2["X"]
        y1 = synthetic_data_level_2["y"]
        X1_res, y1_res = self.resampler.fit_resample(X1, y1)
        self.model_level_1 = clone(self.base_model)
        self.model_level_1.fit(X1_res, y1_res)
        return self

    def predict(self, X):
        """
        Predict using the hierarchical model (level 1 -> level 2).

        :param X: Feature matrix for prediction
        :return: List of predicted labels from level 2 classifiers
        """
        # Level 1 prediction: Classify into a higher-level category (label)
        pred_level_1 = self.model_level_1.predict(X)
        predictions = []
        # For each sample, predict at level 2 based on level 1 predictions
        for xi, label in zip(X, pred_level_1):
            model = self.models_level_2.get(label)  # Get the appropriate model for the predicted level 1 label
            if model is None:
                raise ValueError(f"No model found for level 2 under label '{label}'")
            # Predict with the appropriate level 2 model
            pred = model.predict([xi])[0]
            predictions.append(pred)
        return predictions