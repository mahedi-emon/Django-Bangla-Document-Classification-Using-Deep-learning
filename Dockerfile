FROM python:3.11-slim

# System deps (important for ML)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (cache optimization)
COPY requirements.txt .

# Allow deprecated sklearn package (required by bnltk)
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Upgrade pip and install requirements (suppress root warning)
RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy project code only (models will be downloaded from Hugging Face at runtime)
COPY bangla_classifier_django/ bangla_classifier_django/
COPY classifier/ classifier/
COPY static/ static/
COPY manage.py .

# Static collect
RUN python manage.py collectstatic --noinput

# Run server with increased timeout for Deep Learning inference
CMD ["gunicorn", "bangla_classifier_django.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "120"]