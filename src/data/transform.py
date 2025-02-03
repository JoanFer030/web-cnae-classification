import os
import re
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import open_csv, save_csv

########################################################
###                   FILTER DATA                    ###
########################################################
def filter_by_availability(data: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the data by its website availability.
    """
    available["nif"] = available["BVD_ID"].apply(lambda x: x[2:])
    tqdm.pandas(desc = "Filtering by web availability", postfix=None)
    filtered = pd.merge(data, available, on = "nif", how = "inner").progress_apply(lambda x: x)
    filtered = filtered[data.columns]
    return filtered

########################################################
###                   FORMAT DATA                    ###
########################################################
def edit_columns(data: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """
    Renames columns and deletes unnecessary columns.
    """
    keep_cols = [i for i, n in enumerate(column_names) if n]
    data = data.iloc[:, keep_cols]
    data.columns = [name for name in column_names if name]
    return data

def format_str(value):
    """
    Converts the input value into a string.
    """
    if isinstance(value, str):
        value = value.replace('"', "")
        value = value.replace("'", "")
    return str(value)

def format_int(value: str):
    """
    Converts the input value into a integer.
    """
    try:
        return int(value)
    except:
        return -1

def format_url(value: str) -> str:
    """
    Extracts the url from the input value, in case there is one.
    """
    matchs = re.findall(r"^(www\.[a-zñA-ZÑ0-9.-]+\.[a-zA-Z]{2,})", value)
    if matchs:
        return matchs[0]
    return ""

def format_list(value: str) -> list:
    """
    Converts the input string to a list, where 
    its values is converted to the subtype indicated.
    """
    value = str(value)
    if value == "nan":
        return []
    if "[" in value:
        value = value[1:-1]
        items = value.split(", ")
        return [code[1:-1] for code in items]
    else:
        return [value]

def format_columns(data: pd.DataFrame, column_types: list[str]) -> pd.DataFrame:
    """
    Gives the data the format indicated in the column_types list.
    """
    for name, col_type in tqdm(list(zip(data.columns, column_types)), desc = "Formatting data"):
        if col_type == "str":
            data[name] = data[name].apply(format_str)
        elif col_type == "int":
            data[name] = data[name].apply(format_int)
        elif col_type == "url":
            data[name] = data[name].apply(format_url)
        elif col_type == "list":
            data[name] = data[name].apply(format_list)
        else:
            raise ValueError("Invalid column type.")
    return data

########################################################
###                  MAIN FUNCTIONS                  ###
########################################################
def format_data(config: dict):
    base_path = config["data"]["base_path"]
    sabi_path = base_path + config["data"]["merged_sabi"]
    save_path = base_path + config["data"]["formatted_sabi"]
    column_names = config["data"]["column_names_sabi"]
    column_types = config["data"]["column_types_sabi"]
    sabi = open_csv(sabi_path)
    sabi = edit_columns(sabi, column_names)
    sabi = format_columns(sabi, column_types)
    save_csv(sabi, save_path)

def filter_data(config: dict):
    base_path = config["data"]["base_path"]
    sabi_path = base_path + config["data"]["formatted_sabi"]
    save_path = base_path + config["data"]["filtered_sabi"]
    available_path = base_path + config["data"]["available_api"]
    sabi = open_csv(sabi_path)
    available = open_csv(available_path)
    sabi = filter_by_availability(sabi, available)
    save_csv(sabi, save_path)

def transform_data(config: dict):
    format_data(config)
    filter_data(config)