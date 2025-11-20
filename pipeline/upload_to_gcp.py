import os
import dotenv
from google.cloud import storage

dotenv.load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCATION = "us-east4"

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

def upload_to_gcp(file_path: str, destination_blob_name: str) -> None:
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}")

targets = ["pH", "SOM"]
for target in targets:
    file_path = f"data/batches/{target}_regression_gemini.jsonl"
    destination_blob_name = f"batch_inputs/{target}_regression_gemini.jsonl"
    upload_to_gcp(file_path, destination_blob_name)
    print(f"gcs uri: gs://{BUCKET_NAME}/{destination_blob_name}")

print("Uploaded batch inference inputs to GCS")