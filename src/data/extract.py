import os
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import open_excel, save_csv

########################################################
###                  EXTRACT CNAE                    ###
########################################################
def create_cnae(cnae_path: str, mapping_path: str):
    """
    Extracts the different the codes and labels for each level
    (1-4) for all the CNAE codes. 
    """
    raw_data = open_excel(cnae_path)
    level_data = ["", "", "", "", "", ""]
    data = [["code", "level_1_code", "level_1_label", "level_2_code", "level_2_label", 
             "level_3_code", "level_3_label", "level_4_label"]]
    for row in tqdm(raw_data[1:], desc = "Processing CNAE data"):
        code, _, label = row
        if len(code) == 1:
            level_data[0] = code
            level_data[1] = label
        elif len(code) == 2:
            level_data[2] = code
            level_data[3] = label
        elif len(code) == 3:
            level_data[4] = code
            level_data[5] = label
        else:
            p_row = [code] + level_data + [label]
            data.append(p_row)
    save_csv(data, mapping_path, row_type = "list")

########################################################
###                  EXTRACT SABI                    ###
########################################################
def format_row(rows: list[list]) -> list:
    """
    Processes each batch of rows to obtain one single row.
    """
    processed_row = [[] for _ in range(len(rows[0]))]
    for row in rows:
        for i, value in enumerate(row):
            if value:
                processed_row[i].append(value)
    final_row = [] 
    for values in processed_row:
        l = len(values)
        if l == 0:
            value = ""
        elif l == 1:
            value = values[0]
        else:
            value = values
        final_row.append(value)
    return final_row

def extract_values(line: str) -> tuple[bool, list]:
    """
    Extracts data values from the received raw line.
    """
    values = line.split('";"')
    processed_line = []
    for value in values:
        temp = value.replace('"', "")
        processed_line.append(temp)
    if processed_line[0]:
        return True, processed_line
    return False, processed_line

def load_sabi_file(path: str):
    """
    Loads the file and processes it to obtain the required format.
    """
    with open(path, "r", encoding = "utf-16") as file:
        lines = file.read().splitlines()
    memory = [extract_values(lines[0])[1]]
    data = []
    for line in lines[1:]:
        new, processed_line = extract_values(line)
        if new:
            row = format_row(memory)
            data.append(row)
            memory = []
        memory.append(processed_line)
    return data

def create_sabi(folder_path: str, save_path: str) -> list[list]:
    """
    Lists the directory and processes each file.
    """
    files = os.listdir(folder_path)
    data = []
    for i, file in enumerate(tqdm(files, desc = "Processing SABI files")):
        file_path = folder_path + file
        file_data = load_sabi_file(file_path)
        if i > 0:
            file_data = file_data[1:]
        data.extend(file_data)
    save_csv(data, save_path, row_type = "list")
    return data

########################################################
###                  MAIN FUNCTIONS                  ###
########################################################
def extract_cnae(config: dict):
    base_path = config["data"]["base_path"]
    cnae_path = base_path + config["data"]["raw_cnae"]
    save_path = base_path + config["data"]["processed_cnae"]
    create_cnae(cnae_path, save_path)

def extract_sabi(config: dict):
    base_path = config["data"]["base_path"]
    folder_path = base_path + config["data"]["raw_sabi"]
    save_path = base_path + config["data"]["merged_sabi"]
    create_sabi(folder_path, save_path)

def extract_data(config: dict):
    extract_sabi(config)
    extract_cnae(config)