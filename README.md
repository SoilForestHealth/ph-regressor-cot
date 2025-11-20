<div align="center">
<h1>pH Regression using CoT</h1>
</div>

## Aim

- To predict the value of pH using remote sensing indices, topography, and sampling data.
  
- This repository aims to experiment with whether pre-trained knowledge from large language models helps predict pH in a cold-start setting.

## Environment Setup

- Clone this repository.
  
```bash
git clone https://github.com/Gaurav0502/ph-regressor-cot.git
```

- Install all packages in the ```requirements.txt``` file.

```bash
pip install -r requirements.txt
```

- Ensure the directory structure is as follows:

```bash
.
├── README.md
├── data
│   ├── batches
│   │   ├── batch_outputs_gemini_prediction-model-2025-11-20T01_34_10.422908Z_predictions.jsonl
│   │   └── pH_regression_gemini.jsonl
│   ├── folds
│   │   └── data_pH.json
│   └── grid.csv
├── pipeline
│   ├── batch.py
│   ├── batch_inference.py
│   ├── evaluator.py
│   ├── preprocessor.py
│   ├── prompt.py
│   └── upload_to_gcp.py
├── .env
├── requirements.txt
└── .gitignore
```

In the above directory structure:
- the `*_regression_gemini.jsonl` and `data_*.json`  will be created automatically after `pipeline/preprocessor.py` is executed.
- the `batch_outputs_*_predictions.jsonl` will be created by GCP after your batch inference task finishes executing. (This file can be downloaded from the GCP Bucket)

- Install `gcloud` CLI using the archives from <a href="https://docs.cloud.google.com/sdk/docs/downloads-versioned-archives">Google</a>. Execute the following commands:

```bash
gcloud init
gcloud auth application-default login
```
This will ensure you are authenticated, and this information will be stored locally.

- Create a `.env` file and keep two secrets in it: `PROJECT_ID` and `BUCKET_NAME`. If you are using the API key approach to authenticate, then it goes in this file.

```bash
touch .env
```
Your `.env` file should look as follows:

```txt
PROJECT_ID=<YOUR GCP PROJECT ID>
BUCKET_NAME=<YOUR GCP BUCKET NAME>
```
- Execute the following Python script to create the batches.

```python
import pandas as pd
import numpy as np
from pipeline.preprocessor import BatchCreator

df = pd.read_csv("data/grid.csv")
df.rename(columns={'Organic.Matter....': 'SOM'}, inplace=True)
targets = ["pH", "SOM"]
for target in targets:
    batch_creator = BatchCreator(df.copy(), target)
    batch_creator.create_batch()
    batch_creator.save_batches_as_jsonl()
```

- Execute the commands to push the batches to the GCP bucket. (**)

```bash

cd pipeline
python3 upload_to_gcp.py

```

- Execute the commands to submit batch inferences to Vertex AI. (**)

```bash
cd pipeline
python3 batch_inference.py
```

- Run the Python code below to evaluate the regression model.

```python
from evaluator import Evaluator

df = pd.read_csv("data/grid.csv")
eval = Evaluator(df=df,
                target="pH",
                folds_file_path="data/folds/data_pH.json",
                predictions_file_path="<YOUR PREDICTION FILE PATH>")
```
- This should execute the code in the repository successfully. If there are problems, you can raise an issue!

** This execution requires billing enabled on GCP.

**Note:** When I executed these batches in Vertex AI (`gemini-2.5-flash`), I observed an issue with `maxOutputTokens`. The internal limit is 65,536 tokens. In a few cases, batches were finished due to exceeding the token limit, and no output was returned as the model was interrupted. I do check for this in `pipeline/evaluation.py` before extracting the final prediction.

## References

1. https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction.ipynb
2. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api


