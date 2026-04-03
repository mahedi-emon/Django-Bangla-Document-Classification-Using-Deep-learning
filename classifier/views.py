from django.shortcuts import render
from .ml.predict import predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64


# ===============================
# BAR CHART (STREAMLIT-LIKE)
# ===============================
def plot_bar(classes, probs, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(classes, probs)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.invert_yaxis()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===============================
# MAIN VIEW
# ===============================
def index(request):
    context = {}

    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        context["text"] = text

        # ===============================
        # WORD COUNT VALIDATION (LIKE STREAMLIT)
        # ===============================
        if len(text.split()) < 20:
            context["warning"] = "⚠️ Please provide at least 20 Bangla words."
            return render(request, "classifier/index.html", context)

        # ===============================
        # RUN MODEL
        # ===============================
        result = predict(text)

        # ===============================
        # FINAL RESULT
        # ===============================
        context.update({
            "final_label": result["final_label"],
            "confident": result["confident"],

            # MODEL LABELS
            "bilstm_label": result["bilstm_label"],
            "bert_label": result["bert_label"],
            "ensemble_label": result["ensemble_label"],

            # MODEL CONFIDENCE %
            "bilstm_prob": result["bilstm_prob"],
            "bert_prob": result["bert_prob"],
            "ensemble_prob": result["ensemble_prob"],

            # CHARTS
            "bilstm_chart": plot_bar(
                result["classes"], result["bilstm_probs"], "Bi-LSTM"
            ),
            "bert_chart": plot_bar(
                result["classes"], result["bert_probs"], "Bangla-BERT"
            ),
            "ensemble_chart": plot_bar(
                result["classes"], result["ensemble_probs"], "Hybrid Ensemble"
            ),

            # TABLE DATA
            "rows": zip(
                result["classes"],
                result["bilstm_probs"],
                result["bert_probs"],
                result["ensemble_probs"],
            ),
        })

    return render(request, "classifier/index.html", context)
