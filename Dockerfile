FROM python:3.11-slim

# System deps (important for ML + Git LFS)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy full project (including .git folder)
COPY . .

# Pull actual LFS files using the .git folder
RUN git lfs install && git lfs pull

# Allow deprecated sklearn package (required by bnltk)
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Static collect
RUN python manage.py collectstatic --noinput

# Run server
CMD ["gunicorn", "bangla_classifier_django.wsgi:application", "--bind", "0.0.0.0:8000"]