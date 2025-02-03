import os
import re
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import list_webs, open_web, save_web

########################################################
###                    PROCESS WEBS                  ###
########################################################
def clean_content(content: str) -> str:
    """
    Converts text to lowercase and removes 
    non-alphanumeric characters, leaving only
    words and numbers separated by spaces.
    """
    content = content.lower()
    words = re.findall(r"\b[\wáéíóúüñ]+\b", content)
    clean = " ".join(words)
    return clean

def process_webs(original_path: str, save_path: str):
    """
    Gets the list of all the webs stored and process 
    and stores them again.
    """
    webs = 0
    webs_list = list_webs(original_path)
    for i in tqdm(range(len(webs_list)), desc = "Processing web content"):
        nif, web_path = webs_list[i]
        content = open_web(web_path)
        processed = clean_content(content)
        if processed:
            webs += 1
            save_web(processed, nif, save_path)
    print(f"Processed {webs:,} web sites in {save_path}")
    print(f"{len(webs_list) - webs:,} went wrong")

########################################################
###                  MAIN FUNCTIONS                  ###
########################################################
def process_webs_content(config: dict):
    base_path = config["features"]["base_path"]
    original_path = base_path + config["features"]["original"]
    processed_path = base_path + config["features"]["processed"]
    process_webs(original_path, processed_path)