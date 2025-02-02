import os
import sys
import yaml
import pandas as pd
import argparse
from tqdm import tqdm
tqdm.pandas()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data import open_csv, save_csv

cnae_mapping = open_csv("/home/jfernav/nas/web-nace-classification/data/processed/cnae_mapping.csv")

def get_code_by_level(code: int, level: int) -> str:
    row = cnae_mapping[cnae_mapping["code"] == code]
    if len(row) == 0:
        return "nan"
    if 1 <= level <= 3:
        value = row[f"level_{level}_code"].values[0]
    elif level == 4:
        value = code
    else:
        raise ValueError("Non existing level.")
    return str(value)

def create_balanced(companies: pd.DataFrame, level: int, number: int):
    column = "label"
    companies[column] = companies["primary_cnae"].progress_apply(get_code_by_level, level = level)
    if number is None:
        number = companies[column].value_counts().median()
    balanced_df = pd.DataFrame()
    skipped = []
    for value in tqdm(companies[column].unique(), desc = f"Sampling {number} companies by category"):
        if len(companies[companies[column] == value]) < number:
            skipped.append(value)
            continue
        sampled_rows = companies[companies[column] == value].sample(n = number,
                                                                    replace = False,
                                                                    random_state = 1909)
        balanced_df = pd.concat([balanced_df, sampled_rows], ignore_index=True)
    print(f"Categories {', '.join(skipped)} have been skipped.")
    return balanced_df

def process_balanced(balanced: pd.DataFrame):
    processed = balanced[["nif", "name", "redirect_url", "label", "primary_cnae", "secondary_cnae"]]
    return processed

def create_balanced_dataset(companies_path: str, save_path: str, level: int, number: int):
    companies = open_csv(companies_path)
    balanced = create_balanced(companies, level, number)
    processed = process_balanced(balanced)
    save_csv(processed, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Create CNAE Mapping.")
    parser.add_argument("-i", "--input_path", type = str, help = "Relative path to the file containing all the available companies of SABI.", required = True)
    parser.add_argument("-s", "--save_path", type = str, help = "Relative path to save the balanced dataset CSV file.")
    parser.add_argument("-l", "--level", type = int, help = "Level of the CNAE of which is the data wanted", required = True)
    parser.add_argument("-n", "--number_companies", type = int, help = "Number of companies of each category. -1 for minimum number of each category.", required = True)
    args = parser.parse_args()
    with open("config/parameters.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    base_path = config["data_base_path"]
    companies_path = base_path + args.input_path
    level = args.level
    number = args.number_companies
    if args.save_path:
        save_path = base_path + args.save_path
    else:
        save_path = base_path + f"processed/balanced_level_{level}_companies_{number}.csv"
    create_balanced_dataset(companies_path, save_path, level, number)