import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import load_config
from data.extract import extract_data
from data.transform import transform_data
from data.merge import merge_data

def main(config_path):
    config = load_config(config_path)

    print("\nData Extraction Process Initiated")
    print("-"*60)
    extract_data(config)

    print("\nData Transformation Process Initiated")
    print("-"*60)
    transform_data(config)

    print("\nData Merge Process Initiated")
    print("-"*60)
    merge_data(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the data processing pipeline, including data extraction, transformation, merge, and storage."
    )
    parser.add_argument(
        "--config",
        type     = str,
        required = True,
        help     = "Path to the configuration file (YAML format)."
    )
    args = parser.parse_args()
    main(args.config)