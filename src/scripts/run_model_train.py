import os
import sys
import time
import pickle
import warnings
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data import Data
from utils.files import load_config
from models.hierarchical_level2 import HierarchicalClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

def load_data(config: dict):
    data = Data(config)
    max_level1 = config["models"]["max_size_level1"]
    max_level2 = config["models"]["max_size_level2"]
    data = data.load_hierarchical_training_test_data(
        (max_level2, max_level1),
        0.15,
        ["T", "O", "U"],
        preprocess     = "pooling",
        type           = "mean",
        min_amount     = 7,
        test_original  = True,
        return_test_df = True,
        verbose        = False
    )
    return data

def train_model(train_data):
    print("Setting up Hierarchical Classifier...")
    smote = SMOTE()
    model = SVC(kernel = "linear", C = 1)
    hierarchical_model = HierarchicalClassifier(
        base_model = model, 
        resampler  = smote)
    print("Training hierarchical model...")
    t0 = time.time()
    hierarchical_model.fit(train_data)
    print(f"Training time: {time.time() - t0:.2f} s")
    return hierarchical_model

def save_model_data(config: dict, model, test_data):
    model_path = config["models"]["base_path"] + config["models"]["level_2"]
    print(f"Model saved at: {model_path}")
    with open(model_path,"wb") as file:
        pickle.dump(model, file)

    test_path = config["models"]["base_path"] + config["models"]["test_data"]
    print(f"Test data saved at: {test_path}")
    with open(test_path,"wb") as file:
        pickle.dump(test_data, file)

def main(config_path):
    config = load_config(config_path)

    print("\nData Loading Process Initiated")
    print("-"*60)
    data = load_data(config)
    train, test, test_df = data

    print("\nModel Training Process Initiated")
    print("-"*60)
    model = train_model(train)

    print("\nModel and Data Saving Process Initiated")
    print("-"*60)
    save_model_data(config, model, (test, test_df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training pipeline, including data loading, model training, and model and data saving."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to the configuration file (YAML format)."
    )
    args = parser.parse_args()
    main(args.config)