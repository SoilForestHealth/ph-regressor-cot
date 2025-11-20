import os
import time
import dotenv

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState

dotenv.load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCATION = "us-east4"
MODEL_NAME = "gemini-2.5-flash"
BASE_OUTPUT_DIR = f"gs://{BUCKET_NAME}/batch_outputs/gemini"

target = "pH"
input_uri = f"gs://{BUCKET_NAME}/batch_inputs/{target}_regression_gemini.jsonl"
output_dir = f"{BASE_OUTPUT_DIR}/results_{target}_{int(time.time())}"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

job = client.batches.create(
    model=MODEL_NAME,
    src=input_uri,
    config=CreateBatchJobConfig(dest=output_dir),
)

print(f"Submitted Job: {job.name}")
print(f"Output base dir: {output_dir}")

completed_states = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

while job.state not in completed_states:
    time.sleep(30)
    job = client.batches.get(name=job.name)
    print(f"{job.name} state: {job.state}")

print(f"Final state: {job.state}")