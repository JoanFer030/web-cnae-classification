import os
import csv
import xlrd
import yaml
import pandas as pd

def open_csv(path: str) -> pd.DataFrame:
    """
    Open the data stored in a csv format.
    """
    df = pd.read_csv(path,
                     dtype=str)
    return df

def save_csv(data, save_path: str, row_type = "dict"):
    """
    Saves the data in a csv format.

    """
    directory_path = os.path.dirname(save_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    if isinstance(data, pd.DataFrame):
        data.to_csv(save_path, index = False, quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
        print(f"Saved {len(data):,} rows in {save_path}")
    else:
        with open(save_path, "w", encoding = "utf-8") as file:
            if row_type == "dict":
                writer = csv.writer(file, quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
                header = list(data[0].keys())
                writer.writerow(header)
                for row in data[1:]:
                    r_data = list(row.values())
                    writer.writerow(r_data)
                print(f"Saved {len(data)-1:,} rows in {save_path}")
            elif row_type == "list":
                writer = csv.writer(file, quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
                writer.writerows(data)
                print(f"Saved {len(data):,} rows in {save_path}")
            else:
                raise ValueError("Incorrect row type. It should be 'dict' or 'list'") 

def open_excel(path: str) -> list[list]:
    wb = xlrd.open_workbook(path)
    sheet = wb.sheet_by_index(0)
    n_row = sheet.nrows
    n_col = sheet.ncols
    data = []
    for r in range(n_row):
        row = []
        for c in range(n_col):
            cell = sheet.cell_value(rowx = r, colx = c)
            row.append(cell)
        data.append(row)
    return data

def load_config(config_path):
    """
    Load the YAML configuration file.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)