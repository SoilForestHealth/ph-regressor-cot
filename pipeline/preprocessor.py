# module used for data preprocessing
import pandas as pd
import numpy as np
import json

# module used for cross-validation
from sklearn.model_selection import GroupKFold

# module used for prompt creation
from pipeline.prompt import create_prompt
from pipeline.batch import create_batch
from toon_format import encode, decode

# module used for type hints
from typing import Any

drop_cols_suffix = "nm"

class BatchCreator:
    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self.df = df
        self.target = target

        self.target_to_drop = "SOM" if target == "pH" else "pH"
        self.output_file = f"data/batches/{target}_regression_gemini.jsonl"

    def preprocess(self) -> None:

        cols_to_drop = [col for col in self.df.columns if col.endswith(drop_cols_suffix)]
        self.df.drop(columns = ['Unnamed: 0'] + cols_to_drop + [self.target_to_drop], inplace=True)
        self.df.dropna(inplace=True)

    def create_batch(self) -> None:
        
        self.preprocess()
        groups = self.df['ID'].values
        group_kfold = GroupKFold(n_splits=5)
        y = self.df[self.target]
        data_splits = group_kfold.split(self.df, y, groups)
        fold_id = 1
        data_folds = dict()
        self.batches = []

        for train_index, test_index in data_splits:
            
            train_df = encode(self.df.iloc[train_index].to_dict(orient='records'))
            y_train = y.iloc[train_index].values
            test_df = self.df.iloc[test_index]
            test_df.drop(columns = [self.target], inplace=True)
            train_summary = encode(self.df.iloc[train_index].describe().to_dict(orient='records'))
            batch_id = 1
            data_folds[f"fold_{fold_id}"] = {
                "train_index": train_index.tolist(),
                "test_index": test_index.tolist()
            }

            for i in test_df.iterrows():
                request_id = f"fold_{fold_id}_batch_{batch_id}"
                test_sample = encode(i[1].to_dict())
                prompt = create_prompt(train_df, train_summary, self.target, y_train, test_sample)
                batch = create_batch(request_id, prompt)
                self.batches.append(batch)
                batch_id += 1
            fold_id += 1
        
        json.dump(data_folds, open(f"data/folds/data_{self.target}.json", "w"))
    
    def save_batches_as_jsonl(self):

        with open(self.output_file, "w") as f:
            for batch in self.batches:
                f.write(json.dumps(batch) + "\n")