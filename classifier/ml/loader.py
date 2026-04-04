import os
import pickle
import logging
import joblib
import torch
from tensorflow.keras.saving import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ===============================
# HUGGING FACE REPO CONFIG
# ===============================
HF_REPO_ID = "mahedi6107/bangla-news-classifier"
BERT_SUBFOLDER = "bangla_bert_model"

# ===============================
# LAZY LOADING CACHE
# ===============================
_cache = {}


def _download_from_hf(filename):
    """Download a file from Hugging Face Hub (auto-cached)."""
    logger.info(f"Downloading {filename} from Hugging Face...")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
    )
    logger.info(f"Downloaded to: {path}")
    return path


def _safe_load_pkl(filename):
    """Download .pkl from HF and load with joblib, fallback to pickle."""
    path = _download_from_hf(filename)
    try:
        return joblib.load(path)
    except Exception as e:
        logger.warning(f"joblib.load failed for {filename}: {e}. Trying pickle fallback...")
        with open(path, "rb") as f:
            return pickle.load(f)


def get_device():
    if "device" not in _cache:
        _cache["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _cache["device"]


# ===============================
# Bi-LSTM LOADERS
# ===============================
def get_bilstm_model():
    if "bilstm_model" not in _cache:
        path = _download_from_hf("bilstm_model.keras")
        _cache["bilstm_model"] = load_model(path, compile=False)
    return _cache["bilstm_model"]


def get_bilstm_tokenizer():
    if "bilstm_tokenizer" not in _cache:
        _cache["bilstm_tokenizer"] = _safe_load_pkl("bilstm_tokenizer.pkl")
    return _cache["bilstm_tokenizer"]


def get_label_encoder():
    if "label_encoder" not in _cache:
        _cache["label_encoder"] = _safe_load_pkl("bilstm_label_encoder.pkl")
    return _cache["label_encoder"]


# ===============================
# BERT LOADERS
# ===============================
def get_bert_tokenizer():
    if "bert_tokenizer" not in _cache:
        logger.info(f"Loading BERT tokenizer from HF: {HF_REPO_ID}/{BERT_SUBFOLDER}")
        _cache["bert_tokenizer"] = AutoTokenizer.from_pretrained(
            HF_REPO_ID, subfolder=BERT_SUBFOLDER
        )
    return _cache["bert_tokenizer"]


def get_bert_model():
    if "bert_model" not in _cache:
        logger.info(f"Loading BERT model from HF: {HF_REPO_ID}/{BERT_SUBFOLDER}")
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO_ID, subfolder=BERT_SUBFOLDER
        )
        model.to(get_device())
        model.eval()
        _cache["bert_model"] = model
    return _cache["bert_model"]
