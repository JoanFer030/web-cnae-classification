import os
import sys
import random
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import load_config, open_csv

class Preprocess:
    def __init__(self, dim: int):
        self._dim = dim
        self._preprocess_funcs = {
            "raw": self._raw_embedding,
            "truncate": self._truncate_embedding,
            "pooling":{
                "max": self._pooling_max_embedding,
                "mean": self._pooling_mean_embedding,
                "sum": self._pooling_sum_embedding,
            },
        }

    def get_function(self, input_dict: dict):
        if "preprocess" in input_dict:
            preprocess_func = self._preprocess_funcs.get(input_dict["preprocess"], self._raw_embedding)
            if isinstance(preprocess_func, dict):
                if "type" in input_dict and input_dict["type"] in preprocess_func:
                    preprocess_func = preprocess_func[input_dict["type"]]
                    return preprocess_func
                else:
                    raise ValueError(f"The {input_dict['preprocess']} type ('type') must be a string: {', '.join(list(preprocess_func.keys()))}.")
            else:
                return preprocess_func
        else:
            print("It has not been especified any preprocessing, the raw embedding will be loaded.")
            return self._raw_embedding

    def _raw_embedding(self, embedding: list[float]):
        return embedding

    def _truncate_embedding(self, embedding: list[float]):
        n_embedding = embedding[:self._dim]
        return n_embedding   

    def _split_vector(self, vector, dim):
        return vector.reshape(-1, dim)

    def _pooling_max_embedding(self, embedding: list[float]):
        r_embedding = self._split_vector(embedding, self._dim)
        n_embedding = np.max(r_embedding, axis = 0)
        return n_embedding
    
    def _pooling_mean_embedding(self, embedding: list[float]):
        r_embedding = self._split_vector(embedding, self._dim)
        n_embedding = np.mean(r_embedding, axis = 0)
        return n_embedding
    
    def _pooling_sum_embedding(self, embedding: list[float]):
        r_embedding = self._split_vector(embedding, self._dim)
        n_embedding = np.sum(r_embedding, axis = 0)
        return n_embedding 

class Data:
    def __init__(self, config):
        if isinstance(config, str):
            config = load_config(config)
        elif isinstance(config, dict):
            pass
        else:
            raise ValueError("Invalid input.")
        sabi_path = config["data"]["base_path"] + config["data"]["processed_sabi"]
        cnae_path = config["data"]["base_path"] + config["data"]["processed_cnae"]
        dimensions = config["features"]["embedding_dimension"]
        self._embedding_path = config["features"]["base_path"] + config["features"]["embeddings"]
        self._distribution_path = config["data"]["base_path"] + config["data"]["processed_distribution"]
        self._load_cnae(cnae_path)
        self._load_sabi(sabi_path)
        self._preprocess = Preprocess(dimensions)

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
        self.sabi = dataframe

    def _load_cnae(self, path: str) -> dict:
        cnae_df = open_csv(path)
        cnae = {}
        for _, row in cnae_df.iterrows():
            code = row["code"]
            cnae[code] = {"level_1": (row["level_1_code"], row["level_1_label"]),
                          "level_2": (row["level_2_code"], row["level_2_label"]),
                          "level_3": (row["level_3_code"], row["level_3_label"]),
                          "level_4": (code,                row["level_4_label"])}
        self._cnae = cnae
    
    def load_distribution(self, level: int = 2) -> pd.DataFrame:
        if level not in [1, 2]:
            raise ValueError("The level must be an integer between 1 and 2.")
        dtypes = {"main_activity": str, 
                  "description": str,
                  "total": int}
        distribution = pd.read_csv(self._distribution_path, sep = ";", converters = dtypes, encoding = "latin-1")
        if level == 2:
            return distribution
        distribution["main_activity_1"] = distribution["main_activity"].apply(lambda x: self._get_cnae_data(x, level=level, value="label"))
        distribution["description_1"] = distribution["main_activity"].apply(lambda x: self._get_cnae_data(x, level=level, value="description"))
        grouped_distribution = distribution.groupby(["main_activity_1", "description_1"], as_index=False).agg({"total": "sum"})
        grouped_distribution.rename(columns={
            "main_activity_1": "main_activity",
            "description_1": "description"
        }, inplace=True)
        return grouped_distribution
        
    def _get_embedding_availability(self, nif_list: list[str]) -> list[bool]:
        nif_set = set(nif_list)
        self._embedding_paths = {}
        available_nifs = {file.name[:-4]: file.path for file in os.scandir(self._embedding_path) if file.name[:-4] in nif_set}
        self._embedding_paths.update(available_nifs)
        return [nif in available_nifs for nif in nif_list]

    def _get_cnae_data(self, code: str, level: int, value: str = "label"):
            if value == "label":
                i = 0
            elif value in ["desc", "description"]:
                i = 1
            else:
                raise ValueError("The value type has to be 'label' or 'description'.")
            if len(code) == 4:
                temp = self._cnae.get(code, "")
            else:
                temp = ""
                for key, value in self._cnae.items():
                    if key.startswith(code):
                        temp = value
                        break
            if not temp:
                return pd.NA
            return temp[f"level_{level}"][i]

    def _get_main_label(self, nif: str, level: int) -> str:
        company = self.dataframe[self.dataframe["nif"] == nif]
        if len(company) == 0:
            raise ValueError("The company does not exist.")
        if level > 4:
            raise ValueError("The level must be an integer between 1 and 4.")
        primary_cnae = str(company["primary_cnae"])
        label = self._get_cnae_data(code = primary_cnae, level = level, value = "label")
        return label

    def _get_embedding(self, path: str, preprocess) -> list[float]:
        try:
            embedding = np.load(path)
            processed_embedding = preprocess(embedding)
            return processed_embedding
        except:
            print(f"The file '{path}' does not exist.")

    def load_with_label(self, level = 1):
        if isinstance(level, int):
            if level < 0 or level > 4:
                raise ValueError("The level must be an integer between 1 and 4.")
            label_dataframe = self.sabi.copy()
            label_dataframe["label"] = label_dataframe["primary_cnae"].apply(self._get_cnae_data, level = level, value = "label")
            label_dataframe["description"] = label_dataframe["primary_cnae"].apply(self._get_cnae_data, level = level, value = "description")

        elif isinstance(level, list):
            level_list = level
            for level in level_list:
                if not isinstance(level, int) or (level < 0 or level > 4):
                    raise ValueError("The level must be an integer between 1 and 4.")
            label_dataframe = self.sabi.copy()
            for level in level_list:
                label_dataframe[f"label_{level}"] = label_dataframe["primary_cnae"].apply(self._get_cnae_data, level = level, value = "label")
                label_dataframe[f"description_{level}"] = label_dataframe["primary_cnae"].apply(self._get_cnae_data, level = level, value = "description")
        return label_dataframe
    
    def random_sample(self, n: int, level) -> list[tuple[list[float], tuple[str]]]:
        """
        Generates a random sample of `n` rows from the dataset and retrieves their corresponding embeddings and labels.

        This function selects a random sample from the dataset (`self.sabi`), retrieves the embedding for each sample,
        and collects the labels based on the specified `level` parameter. The resulting sample consists of 
        tuples where each tuple contains an embedding and its corresponding label(s).

        Parameters:
        -----------
        n : int
            The number of random samples to retrieve.
        
        level : list or int
            If `level` is a list, multiple labels are extracted.
            If `level` is an integer, a single label is extracted.

        Returns:
        --------
        list[tuple[list[float], tuple[str]]]
            A list of tuples, where each tuple contains:
            - A list of floats representing the embedding of the sample.
            - A tuple of strings representing the label(s) associated with the sample.
        """
        sample = []
        level_data = self.load_with_label(level = level)
        level_data = level_data[level_data["available"] == True].reset_index()
        random_sample = level_data.sample(n = n)
        prep_func = self._preprocess.get_function({"preprocess": "raw"})
        for _, row in random_sample.iterrows():
            nif = row["nif"]
            path = self._embedding_paths.get(nif, "")
            if not path:
               continue
            embedding = self._get_embedding(path, prep_func)
            if isinstance(level, list):
                label = []
                for l in level:
                    label.append(row[f"label_{l}"])
                label = tuple(label)
            elif isinstance(level, int):
                label = row["label"]
            sample.append((embedding, label))
        return sample

    def _load_data_from_df(self, df: pd.DataFrame, preprocess_func) -> tuple[list[list[float]], list[str]]:
        X = []
        y = []
        labels = [name for name in df.columns if name.startswith("label")]
        for _, row in df.iterrows():
            nif = row["nif"]
            path = self._embedding_paths.get(nif, "")
            if not path:
                continue
            embedding = self._get_embedding(path, preprocess_func)
            if len(labels) > 1:
                label = []
                for l in labels:
                    label.append(row[l])
                label = tuple(label)
            else:
                label = row[labels[0]]
            X.append(embedding)
            y.append(label)
        return X, y

    def load_all(self, level: int = 1) -> tuple[list[list[float]], list[str]]:
        temp = self.load_with_label(level)
        companies = temp[temp["available"] == True].reset_index()
        prep_func = self._preprocess.get_function({"preprocess": "raw"})
        X, y = self._load_data_from_df(companies, prep_func)
        return X, y

    def load_sample(self, level, balanced: bool = False, **kwargs) -> tuple[list[list[float]], list[str]]:
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
        seed = kwargs.get("seed", int(random.random()*1000))
        temp = self.load_with_label(level)
        temp = temp[temp["available"] == True].reset_index()
        sample = pd.DataFrame()
        if isinstance(level, list):
            max_detail = max(level)
            label_name = f"label_{max_detail}"
        elif isinstance(level, int):
            label_name = "label"
        skipped = []
        if balanced:
            for label in temp[label_name].unique():
                subset = temp[temp[label_name] == label]
                if len(subset) < min_amount:
                    skipped.append(label)
                    continue
                subset_sample = subset.sample(n            = min_amount, 
                                              replace      = False, 
                                              random_state = seed)
                sample = pd.concat([sample, subset_sample], ignore_index = True)
        else:
            for label in temp[label_name].unique():
                subset = temp[temp[label_name] == label]
                amount = int(round((len(subset)/len(temp)) * total, 0))
                print(label, amount)
                if amount < 1:
                    skipped.append(label)
                subset_sample = subset.sample(n            = amount, 
                                              replace      = False, 
                                              random_state = seed)
                sample = pd.concat([sample, subset_sample], ignore_index = True)
        print(f"Categories {', '.join(skipped)} have been skipped.")
        preprocess_func = self._preprocess.get_function(kwargs)
        X, y = self._load_data_from_df(sample, preprocess_func)
        print(f"The embedding and label of {len(X)} companies have been loaded.")
        return X, y

    def load_training_test_data(self, level, train_cat_size: int, test_cat_prop: float, discard: list = [], **kwargs) -> tuple[list[list[float]], list[str]]:
        if not (0 < test_cat_prop < 1):
            raise ValueError("The test size has to be between 0 and 1.")
        seed = kwargs.get("seed", int(random.random()*1000))
        min_amount = kwargs.get("min_amount", 0)
        if "custom_dim" in kwargs:
            initial_dim = self._preprocess._dim
            self._preprocess = Preprocess(kwargs["custom_dim"])
        temp = self.load_with_label(level)
        temp = temp[temp["available"] == True].reset_index()
        if isinstance(level, list):
            max_detail = max(level)
            label_name = f"label_{max_detail}"
        elif isinstance(level, int):
            label_name = "label"
        labels = temp[label_name].unique()
        # Training sample
        train_sample = pd.DataFrame()
        if isinstance(level, list) or level > 1:
            to_skip = [l for l in labels if self._get_cnae_data(l, 1, "label") in discard]
        else:
            to_skip = discard
        for label in labels:
            subset = temp[temp[label_name] == label]
            if label in to_skip:
                continue
            elif len(subset) < ((1 + test_cat_prop) * train_cat_size):
                size = int(len(subset) - (test_cat_prop * train_cat_size))
            else:
                size = train_cat_size
            if size < min_amount:
                print(f"Category {label} has been skipped due to an insufficient number of samples ({len(subset)} samples, at least {int((test_cat_prop * train_cat_size) + 1)} required).")
                to_skip.append(label)
                continue
            subset_sample = subset.sample(n            = size, 
                                          replace      = False, 
                                          random_state = seed)
            train_sample = pd.concat([train_sample, subset_sample], ignore_index = True)
        # Test sample
        test_sample = pd.DataFrame()
        test_cat_size = ((len(labels) - len(to_skip)) * train_cat_size) * test_cat_prop
        for label in labels:
            if label in to_skip:
                continue
            temp_subset = temp[temp[label_name] == label]
            subset = temp_subset[~temp_subset["nif"].isin(train_sample["nif"])]
            amount = int((len(temp_subset)/len(temp)) * test_cat_size)
            if amount < 1:
                continue
            subset_sample = subset.sample(n            = amount, 
                                          replace      = False, 
                                          random_state = seed)
            test_sample = pd.concat([test_sample, subset_sample], ignore_index = True)
        print(f"Categories {', '.join(to_skip)} have been skipped.")
        prep_func = self._preprocess.get_function(kwargs)
        X_train, y_train = self._load_data_from_df(train_sample, prep_func)
        X_test, y_test = self._load_data_from_df(test_sample, prep_func)
        print(f"It has loaded {len(X_train)} companies for training and {len(X_test)} companies for testing, spanning {len(labels) - len(to_skip)} categories.")
        if "custom_dim" in kwargs:
            self._preprocess = Preprocess(initial_dim)
        return (X_train, y_train), (X_test, y_test)

    def load_hierarchical_training_test_data(self, max_level: int, train_amount: tuple[int, int], test_prop: float, to_skip: list, **kwargs):
        if max_level < 2 or max_level > 4:
            raise ValueError("The maximum detail level must be between 2 and 4.")
        seed = kwargs.get("seed", int(random.random()*1000))
        min_amount = kwargs.get("min_amount", 0)

        levels = list(range(1, max_level+1, 1))
        temp = self.load_with_label(levels)
        temp = temp[temp["available"] == True].reset_index()
        test_size = ((len(temp["label_2"].unique())) * train_amount[0]) * test_prop

        train_level_1 = ([], [])
        train_level_2 = {}
        test = ([], [])
        prep_func = self._preprocess.get_function(kwargs)

        for level_1_label in temp["label_1"].unique():
            if level_1_label in to_skip:
                continue
            subset_1 = temp[temp["label_1"] == level_1_label]
            labels_level_2 = subset_1["label_2"].unique()
            default_level_2 = max(train_amount[0], int(min(len(subset_1), train_amount[1])/len(labels_level_2)))
            temp_train_level_2 = ([], [])

            print("Section:", level_1_label)

            for level_2_label in labels_level_2:
                subset_2 = subset_1[subset_1["label_2"] == level_2_label]
                test_amount = max(1, int((len(subset_2)/len(temp)) * test_size))
                if len(subset_2) <= (test_amount + default_level_2):
                    size = int(len(subset_2) - test_amount)
                else:
                    size = default_level_2
                if size < min_amount:
                    print(f"Category {level_2_label} has been skipped due to an insufficient number of samples.")
                    continue

                print("  - Division:", level_2_label)
                print(f"    Subset: {len(subset_2)}  |  Train: {size}  |  Test: {test_amount}  | Default: {default_level_2}")

                train_subset = subset_2.sample(n            = size, 
                                               replace      = False, 
                                               random_state = seed)
                temp_test_subset = subset_2[~subset_2["nif"].isin(train_subset["nif"])]
                if test_amount < 1:
                    continue
                test_subset = temp_test_subset.sample(n            = test_amount, 
                                                      replace      = False, 
                                                      random_state = seed)
                
                X_train, _ = self._load_data_from_df(train_subset, prep_func)
                X_test, _ = self._load_data_from_df(test_subset, prep_func)
                train_level_1[0].extend(X_train)
                train_level_1[1].extend([level_1_label] * len(X_train))
                temp_train_level_2[0].extend(X_train)
                temp_train_level_2[1].extend([level_2_label] * len(X_train))
                test[0].extend(X_test)
                test[1].extend([level_2_label] * len(X_test))
            print(f"  Total: {len(temp_train_level_2[0])}")
            
            train_level_2[level_1_label] = temp_train_level_2

        return train_level_1, train_level_2, test