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

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Static collect
RUN python manage.py collectstatic --noinput

# Run server
CMD ["gunicorn", "bangla_classifier_django.wsgi:application", "--bind", "0.0.0.0:8000"]