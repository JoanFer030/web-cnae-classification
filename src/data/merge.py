import os
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data import open_csv, save_csv

########################################################
###                   REDIRECTION                    ###
########################################################
def replace_original(row):
    """
    In case the redirected url is empty it is replace by the original.
    """
    if str(row["redirect_url"]) == "nan":
        row["redirect_url"] = row["sabi_url"]
    return row

def add_redirected_url(data: pd.DataFrame, redirects: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the redirected url.
    """
    redirects.columns = ["", "id", "sabi_url", "redirect_url"]
    data_redirected = pd.merge(data, redirects, on = "sabi_url", how = "inner")
    data_redirected = data_redirected[list(data.columns) + ["redirect_url"]]
    tqdm.pandas(desc = "Merging redirected URL", postfix=None)
    data_redirected = data_redirected.progress_apply(replace_original, axis = 1)
    data_redirected = data_redirected.drop_duplicates()
    return data_redirected

########################################################
###                  MAIN FUNCTIONS                  ###
########################################################
def merge_sabi_redirects(config: dict):
    base_path = config["data"]["base_path"]
    sabi_path = base_path + config["data"]["filtered_sabi"]
    save_path = base_path + config["data"]["processed_sabi"]
    redirections_path = base_path + config["data"]["redirections"]
    sabi = open_csv(sabi_path)
    redirections = open_csv(redirections_path)
    sabi = add_redirected_url(sabi, redirections)
    save_csv(sabi, save_path)

def merge_data(config: dict):
    merge_sabi_redirects(config)