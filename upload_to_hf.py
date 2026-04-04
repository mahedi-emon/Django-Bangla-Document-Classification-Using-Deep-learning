"""
Upload models to Hugging Face Hub
"""
import os
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from huggingface_hub import HfApi

# ===============================
# Hugging Face Config
# ===============================
HF_USERNAME = "mahedi6107"
REPO_NAME = "bangla-news-classifier"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# ===============================
# Model file paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_UPLOAD = [
    ("models/bangla_bert_model/config.json", "bangla_bert_model/config.json"),
    ("models/bangla_bert_model/model.safetensors", "bangla_bert_model/model.safetensors"),
    ("models/bangla_bert_model/tokenizer.json", "bangla_bert_model/tokenizer.json"),
    ("models/bangla_bert_model/tokenizer_config.json", "bangla_bert_model/tokenizer_config.json"),
    ("models/bangla_bert_model/special_tokens_map.json", "bangla_bert_model/special_tokens_map.json"),
    ("models/bangla_bert_model/vocab.txt", "bangla_bert_model/vocab.txt"),
    ("models/bilstm_model.keras", "bilstm_model.keras"),
    ("models/bilstm_tokenizer.pkl", "bilstm_tokenizer.pkl"),
    ("models/bilstm_label_encoder.pkl", "bilstm_label_encoder.pkl"),
]


def main():
    api = HfApi()

    print(f"Hugging Face Repo: {REPO_ID}")
    print("=" * 50)

    for local_path, hf_path in FILES_TO_UPLOAD:
        full_local_path = os.path.join(BASE_DIR, local_path)

        if not os.path.exists(full_local_path):
            print(f"[SKIP] File not found: {local_path}")
            continue

        file_size_mb = os.path.getsize(full_local_path) / (1024 * 1024)
        print(f"\n[UPLOADING] {local_path} ({file_size_mb:.1f} MB)")

        try:
            api.upload_file(
                path_or_fileobj=full_local_path,
                path_in_repo=hf_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"[OK] {hf_path}")
        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n" + "=" * 50)
    print(f"DONE! Model link: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
