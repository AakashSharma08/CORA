# download_routerbench.py
import pandas as pd
from huggingface_hub import hf_hub_download
import json

print("Downloading RouterBench (0-shot dataset)...")
file_path = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")

print("Loading pickle file with Pandas...")
df = pd.read_pickle(file_path)

# Extract 500 samples
samples = []
for idx, row in df.head(500).iterrows():
    # Convert row to dict, handling any non-serializable objects
    item = row.to_dict()
    # The prompt is likely under a column named 'prompt' or 'instruction'
    samples.append(item)

with open("routerbench_samples.json", "w") as f:
    json.dump(samples, f, default=str)

print(f"Saved {len(samples)} samples to routerbench_samples.json")
if len(samples) > 0:
    print("Keys in each sample:", list(samples[0].keys()))
