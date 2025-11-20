import pandas as pd
import numpy as np
import json
import re

from sklearn.metrics import mean_squared_error, r2_score

class Evaluator:
    def __init__(self, df: pd.DataFrame, target: str, folds_file_path: str, predictions_file_path: str) -> None:

        self.df = df
        self.folds_file_path = folds_file_path
        self.predictions_file_path = predictions_file_path

        self.target = target
        self.target_to_drop = "SOM" if target == "pH" else "pH"

        self.folds = json.load(open(self.folds_file_path))
        self.ground_truth = None
        self.predictions = None
    
    def extract_predictions(self) -> None:

        self.predictions = []

        f = open(self.predictions_file_path, 'r')
        llm_output = []

        for line in f:
            llm_output.append(json.loads(line))
        
        print(f"Total records: {len(llm_output)}")
        
        for i in llm_output:

            if "avgLogprobs" in i["response"]["candidates"][0]:
                prediction_part = i["response"]["candidates"][0]["content"]["parts"][0]["text"].split("Final Prediction:")

                if len(prediction_part) == 1:
                    # output is truncated due to token limit
                    continue

                p = prediction_part[1].split("\n")[0]
                self.predictions.append((i["key"], 
                                         float(re.search(r'\d+\.\d+', p).group())))
        
        print(f"Total predictions: {len(self.predictions)}")

    def __preprocess(self) -> None:
      
        self.df.rename(columns={'Organic.Matter....': 'SOM'}, inplace=True)
        cols_to_drop = [col for col in self.df.columns if col.endswith("nm")]
        self.df.drop(columns=['Unnamed: 0'] + cols_to_drop + ["SOM"], inplace=True)
        self.df.dropna(inplace=True)

    def evaluate(self) -> None:

        self.__preprocess()

        self.ground_truth = dict()
        self.predictions = dict()

        for i in self.predictions:
            
            request_id = i[0]
            fold = "_".join(request_id.split("_")[0:2])
            row_id = int(request_id.split("_")[-1]) - 1

            test_index_pos = self.folds[fold]["test_index"][row_id]
            ground_truth = self.df.iloc[test_index_pos][self.target]
            self.ground_truth[request_id] = ground_truth
            self.predictions[request_id] = i[1]
        
        print("ground truth and predictions extracted...")

        r2_scores = []
        rmse_scores = []

        for request_id in self.ground_truth:
            r2_score = r2_score(self.ground_truth[request_id], self.predictions[request_id])
            mse_score = mean_squared_error(self.ground_truth[request_id], self.predictions[request_id])
            r2_scores.append(r2_score)
            rmse_scores.append(np.sqrt(mse_score))

        print("R2 score and RMSE scores calculated...")
        print(f"R2 score: {np.mean(r2_scores)}")
        print(f"RMSE score: {np.mean(rmse_scores)}")

        return np.mean(r2_scores), np.mean(rmse_scores)