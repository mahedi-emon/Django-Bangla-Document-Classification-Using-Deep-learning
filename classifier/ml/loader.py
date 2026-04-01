import os
import joblib
import torch
from tensorflow.keras.saving import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings

# ===============================
# BASE DIRECTORY (CRITICAL)
# ===============================
BASE_DIR = settings.BASE_DIR

# ===============================
# PATHS (ABSOLUTE, NOT RELATIVE)
# ===============================
BILSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "bilstm_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "bilstm_tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "bilstm_label_encoder.pkl")
BERT_PATH = os.path.join(BASE_DIR, "models", "bangla_bert_model")

# ===============================
# LAZY LOADING CACHE
# ===============================
_cache = {}

def get_device():
    if "device" not in _cache:
        _cache["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _cache["device"]

# ===============================
# Bi-LSTM LOADERS
# ===============================
def get_bilstm_model():
    if "bilstm_model" not in _cache:
        _cache["bilstm_model"] = load_model(BILSTM_MODEL_PATH, compile=False)
    return _cache["bilstm_model"]

def get_bilstm_tokenizer():
    if "bilstm_tokenizer" not in _cache:
        _cache["bilstm_tokenizer"] = joblib.load(TOKENIZER_PATH)
    return _cache["bilstm_tokenizer"]

def get_label_encoder():
    if "label_encoder" not in _cache:
        _cache["label_encoder"] = joblib.load(LABEL_ENCODER_PATH)
    return _cache["label_encoder"]

# ===============================
# BERT LOADERS
# ===============================
def get_bert_tokenizer():
    if "bert_tokenizer" not in _cache:
        _cache["bert_tokenizer"] = AutoTokenizer.from_pretrained(BERT_PATH)
    return _cache["bert_tokenizer"]

def get_bert_model():
    if "bert_model" not in _cache:
        model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
        model.to(get_device())
        model.eval()
        _cache["bert_model"] = model
    return _cache["bert_model"]
