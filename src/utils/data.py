import os
import sys
import random
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import load_config, open_csv

class Data:
    def __init__(self, config_path: str):
        config = load_config(config_path)
        sabi_path = config["data"]["base_path"] + config["data"]["processed_sabi"]
        cnae_path = config["data"]["base_path"] + config["data"]["processed_cnae"]
        self.embedding_path = config["features"]["base_path"] + config["features"]["embeddings"]
        self.embedding_paths = {}
        self.cnae = self._load_cnae(cnae_path)
        self.sabi = self._load_sabi(sabi_path)

    def _load_sabi(self, path: str) -> pd.DataFrame:
        dtypes = {"name": str, 
                  "nif": str, 
                  "province": str, 
                  "sabi_url": str, 
                  "employees": int, 
                  "primary_cnae": str,
                  "secondary_cnae": eval, 
                  "redirect_url": str}
        dataframe = pd.read_csv(path, converters = dtypes)
        dataframe = dataframe[["nif", "name", "employees", "province",
                               "redirect_url", "primary_cnae", "secondary_cnae"]]
        dataframe.columns = ["nif", "name", "employees", "province",
                               "url", "primary_cnae", "secondary_cnae"]
        dataframe["available"] = self._get_embedding_availability(dataframe["nif"].to_list())
        return dataframe

    def _load_cnae(self, path: str) -> dict:
        cnae_df = open_csv(path)
        cnae = {}
        for _, row in cnae_df.iterrows():
            code = row["code"]
            cnae[code] = {"level_1": (row["level_1_code"], row["level_1_label"]),
                          "level_2": (row["level_2_code"], row["level_2_label"]),
                          "level_3": (row["level_3_code"], row["level_3_label"]),
                          "level_4": (code,                row["level_4_label"])}
        return cnae

    def _get_embedding_availability(self, nif_list: list[str]) -> bool:
        nif_availability = [False] * len(nif_list)
        for file in os.scandir(self.embedding_path):
            name = file.name[:-4]
            try:
                idx = nif_list.index(name)
                nif_availability[idx] = True
                self.embedding_paths[name] = file.path
            except:
                pass
        return nif_availability

    def _get_main_label(self, nif: str, level: int) -> str:
        company = self.dataframe[self.dataframe["nif"] == nif]
        if len(company) == 0:
            raise ValueError("The company does not exist.")
        if level > 4:
            raise ValueError("The level must be an integer between 1 and 4.")
        primary_cnae = str(company["primary_cnae"])
        info = self.cnae.get(primary_cnae, "")
        if not info:
            return pd.NA
        label = info[f"level_{level}"][0]
        return label

    def _get_embedding(self, path: str, process: str, len_embedding: int = 1536) -> list[float]:
        try:
            with open(path, "r") as file:
                if process == "none":
                    lines = file.readlines()
                    embedding = [float(value) for value in lines]
                elif process == "first":
                    embedding = []
                    while len(embedding) < len_embedding:
                        embedding.append(float(file.readline()))
                else:
                    print("The preprocess type does not exists.")
            return embedding
        except:
            print(f"The file '{path}' does not exist.")
    
    def load_with_label(self, level: int = 1):
        if level < 0 or  level > 4:
            raise ValueError("The level must be an integer between 1 and 4.")
        def get_label(code: str, level: int):
            temp = self.cnae.get(code, "")
            if not temp:
                return pd.NA
            return temp[f"level_{level}"][0]
        def get_desc(code: str, level: int):
            temp = self.cnae.get(code, "")
            if not temp:
                return pd.NA
            return temp[f"level_{level}"][1]
            
        label_dataframe = self.sabi.copy()
        label_dataframe["label"] = label_dataframe["primary_cnae"].apply(get_label, level = level)
        label_dataframe["description"] = label_dataframe["primary_cnae"].apply(get_desc, level = level)
        return label_dataframe

    def load_all(self, level: int = 1, embedding_process: str = "none") -> tuple[list[list[float]], list[str]]:
        temp = self.load_with_label(level)
        temp = temp[temp["available"] == True].reset_index()
        X = []
        y = []
        for _, row in temp.iterrows():
            nif = row["nif"]
            path = self.embedding_paths.get(nif, "")
            if not path:
               continue
            embedding = self._get_embedding(path, embedding_process)
            label = row["label"]
            X.append(embedding)
            y.append(label)
        return X, y

    def load_sample(self, balanced: bool = False, level: int = 1, embedding_process: str = "none"
                    , **kwargs) -> tuple[list[list[float]], list[str]]:
        if balanced:
            if "min_amount" not in kwargs:
                raise ValueError("Enter the 'min_amount' parameter to sample a balanced dataset.")
            else:
                min_amount = kwargs["min_amount"]
        if not balanced:
            if "total" not in kwargs:
                raise ValueError("Enter the 'total' parameter to sample an unbalanced dataset.")
            else:
                total = kwargs["total"]
        if "seed" in kwargs:
            seed = kwargs["seed"]
        else:
            seed = int(random.random()*1000)
        temp = self.load_with_label(level)
        temp = temp[temp["available"] == True].reset_index()
        sample = pd.DataFrame()
        if balanced:
            skipped = []
            for label in temp["label"].unique():
                subset = temp[temp["label"] == label]
                if len(subset) < min_amount:
                    skipped.append(label)
                    continue
                subset_sample = subset.sample(n            = min_amount, 
                                              replace      = False, 
                                              random_state = seed)
                sample = pd.concat([sample, subset_sample], ignore_index = True)
            print(f"Categories {', '.join(skipped)} have been skipped.")
        else:
            for label in temp["label"].unique():
                subset = temp[temp["label"] == label]
                amount = int(round((len(subset)/len(temp)) * total, 0))
                skipped = []
                if amount < 1:
                    skipped.append(label)
                subset_sample = subset.sample(n            = amount, 
                                              replace      = False, 
                                              random_state = seed)
                sample = pd.concat([sample, subset_sample], ignore_index = True)
            print(f"Categories {', '.join(skipped)} have been skipped.")
        X = []
        y = []
        for _, row in sample.iterrows():
            nif = row["nif"]
            path = self.embedding_paths.get(nif, "")
            if not path:
                continue
            embedding = self._get_embedding(path, embedding_process)
            label = row["label"]
            X.append(embedding)
            y.append(label)
        return X, y