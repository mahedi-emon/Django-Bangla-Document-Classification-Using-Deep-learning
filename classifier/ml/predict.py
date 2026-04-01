import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .loader import (
    get_bilstm_model,
    get_bilstm_tokenizer,
    get_label_encoder,
    get_bert_model,
    get_bert_tokenizer,
    get_device,
)
from .preprocess import clean_article, remove_stopwords

# ===============================
# CONFIDENCE THRESHOLD
# ===============================
CONF_THRESHOLD = 0.15


def predict(text):
    # ===============================
    # Bi-LSTM Prediction
    # ===============================
    t = remove_stopwords(clean_article(text))
    seq = get_bilstm_tokenizer().texts_to_sequences([t])
    pad = pad_sequences(seq, maxlen=400, padding="post")
    bilstm_probs = get_bilstm_model().predict(pad, verbose=0)[0]

    # ===============================
    # Bangla-BERT Prediction
    # ===============================
    inputs = get_bert_tokenizer()(
        clean_article(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(get_device()) for k, v in inputs.items()}

    with torch.no_grad():
        logits = get_bert_model()(**inputs).logits

    bert_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # ===============================
    # Hybrid Ensemble (Soft Voting)
    # ===============================
    ensemble_probs = 0.5 * bilstm_probs + 0.5 * bert_probs

    classes = get_label_encoder().classes_

    # ===============================
    # CONFIDENCE THRESHOLD LOGIC
    # ===============================
    idx_sorted = np.argsort(ensemble_probs)
    top1, top2 = idx_sorted[-1], idx_sorted[-2]

    if ensemble_probs[top1] - ensemble_probs[top2] < CONF_THRESHOLD:
        final_label = f"Ambiguous ({classes[top1]} / {classes[top2]})"
        confident = False
    else:
        final_label = classes[top1]
        confident = True

    # ===============================
    # MAX PROBABILITIES
    # ===============================
    bilstm_max_prob = float(np.max(bilstm_probs))
    bert_max_prob = float(np.max(bert_probs))
    ensemble_max_prob = float(np.max(ensemble_probs))

    # ===============================
    # RETURN RESULTS (FOR DJANGO)
    # ===============================
    return {
        "final_label": final_label,
        "confident": confident,

        "classes": classes.tolist(),

        "bilstm_label": classes[int(np.argmax(bilstm_probs))],
        "bert_label": classes[int(np.argmax(bert_probs))],
        "ensemble_label": classes[int(np.argmax(ensemble_probs))],

        "bilstm_prob": round(bilstm_max_prob * 100, 2),
        "bert_prob": round(bert_max_prob * 100, 2),
        "ensemble_prob": round(ensemble_max_prob * 100, 2),

        "bilstm_probs": bilstm_probs.tolist(),
        "bert_probs": bert_probs.tolist(),
        "ensemble_probs": ensemble_probs.tolist(),
    }
