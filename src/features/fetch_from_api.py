import os
import sys
import json
import requests
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import open_csv, save_web, list_files

########################################################
###                     FETCH WEBS                   ###
########################################################
def fetch_web(company_url: str, api_key: str):
    """
    Performs the GET request with the necessary
    parameters to the API.
    """
    api_url = "http://api.josepdomenech.com/indicators/get-content/"
    query_params = {
       'project': "innov25",
       'url': company_url,
       'date': "2025-01-01",
       'format': "txt",
       'api-key': api_key
       }
    response = requests.get(url = api_url, params = query_params)
    if response.status_code == 200:
        try:
            json_content = json.loads(response.content)
            return json_content.get("content", "")
        except:
            return ""
    return ""

def download_web_content(companies: pd.DataFrame, folder_path: str, api_key: str):
    """
    Downloads all the web pages stored in the companies file, if
    they exists in the API.
    """
    webs = 0
    already_fetch = set([n for n, _ in list_files(folder_path)])
    for idx in tqdm(range(len(companies))):
        row = companies.loc[idx]
        nif, url = row["nif"], row["redirect_url"]
        if nif in already_fetch:
            webs += 1
            continue
        content = fetch_web(url, api_key)
        if content:
            webs += 1
            save_web(content, nif, folder_path)
    print(f"Saved {webs:,} web sites in {folder_path}")
    print(f"{len(companies) - webs:,} went wrong")

########################################################
###                  MAIN FUNCTIONS                  ###
########################################################
def fetch_from_api(config: dict, secrets: dict):
    base_path = config["data"]["base_path"]
    sabi_path = base_path + config["data"]["processed_sabi"]
    base_path = config["features"]["base_path"]
    save_path = base_path + config["features"]["original"]
    api_key = secrets["web_api_key"]
    sabi = open_csv(sabi_path)
    download_web_content(sabi, save_path, api_key)