import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import load_config
from features.fetch_from_api import fetch_from_api
from features.process_web_content import process_webs_content
from features.create_embeddings import create_embeddings

def main(config_path: str, secrets_path: str):
    config = load_config(config_path)
    secrets = load_config(secrets_path)

    print("\nFetching Web Data Process Initiated")
    print("-"*60)
    fetch_from_api(config, secrets)
    
    print("\nProcessing Web Data Initiated")
    print("-"*60)
    process_webs_content(config)
    
    print("\nEmbedding Creation Process Initiated")
    print("-"*60)
    create_embeddings(config, secrets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the text features pipeline, including fetching web content from an API, cleaning the extracted text, and generating text embeddings."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML format)."
    )
    parser.add_argument(
        "--secrets",
        type=str,
        required=True,
        help="Path to the private data file (YAML format) containing API credentials or sensitive information."
    )
    args = parser.parse_args()
    main(args.config, args.secrets)
