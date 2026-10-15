FROM python:3.11-slim

WORKDIR /app

# System deps xgboost's wheel needs at runtime on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
# xgboost's Linux wheel pulls in nvidia-nccl-cu12 (a ~350MB GPU library) as a
# dependency even for plain CPU use. We install everything else normally,
# then install xgboost with --no-deps -- its real runtime needs (numpy, scipy)
# are already satisfied by pandas/scikit-learn above, so this is safe and
# avoids that huge, unnecessary, flaky download.
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    flask gunicorn joblib numpy pandas scikit-learn \
 && pip install --no-cache-dir --no-deps --retries 5 --timeout 120 xgboost

# Only what's needed to SERVE: code + the already-trained artifacts.
# Fraud.csv, eda.py, train_eval.py, eda_output/ etc. are intentionally left out (see .dockerignore).
COPY app.py features.py model_v2.pkl metrics.json ./
COPY static/ ./static/

ENV PORT=5000
EXPOSE 5000

# $PORT is set by most PaaS platforms (Render, Railway, Fly.io); defaults to 5000 locally.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 60 app:app"]
