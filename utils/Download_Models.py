from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from huggingface_hub import snapshot_download
import os

# Specify download directory
model_dir = ""
os.makedirs(model_dir, exist_ok=True)

# Download models
models_to_download = {
    "meta-llama/Llama-2-7b-hf": {
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer
    },
    "Qwen/Qwen2.5-7B": {
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer
    },
    "google-bert/bert-base-chinese": {
        "model_class": AutoModel,
        "tokenizer_class": AutoTokenizer
    }
}

for model_id, classes in models_to_download.items():
    print(f"Starting to download {model_id}...")
    local_path = os.path.join(model_dir, model_id.split("/")[-1])

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
        )
        print(f"{model_id} downloaded successfully, saved to: {local_path}")

    except Exception as e:
        print(f"Error downloading {model_id}: {e}")
        print(f"Attempting to download using model class...")

        # Download directly using model class (alternative)
        try:
            tokenizer = classes["tokenizer_class"].from_pretrained(model_id)
            model = classes["model_class"].from_pretrained(model_id)

            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            print(f"{model_id} downloaded successfully, saved to: {local_path}")
        except Exception as e2:
            print(f"Alternative download method failed: {e2}")