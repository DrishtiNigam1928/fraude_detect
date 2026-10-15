"""Fraud screening API. Run `python train_eval.py --csv Fraud.csv` first to create model_v2.pkl."""
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request, send_from_directory

from features import TYPES  # noqa: F401  (add_features must be importable for unpickling)

BASE = Path(__file__).resolve().parent
MAX_VALUE = 1e12

LABELS = {
    "amount": "Transaction amount",
    "oldbalanceOrg": "Sender balance",
    "oldbalanceDest": "Receiver balance",
    "hour": "Hour of day",
    "amount_ratio": "Share of sender balance moved",
    "is_night": "Made at night",
    "orig_empty": "Sender balance was 0",
    "dest_empty": "Receiver balance was 0",
}


def pretty(feature_name: str) -> str:
    name = feature_name.split("__", 1)[-1]
    if name.startswith("type_"):
        return f"Type is {name[5:]}"
    return LABELS.get(name, name)


def parse_payload(data):
    """Validate and normalise the request body. Raises ValueError(message, field)."""
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object.", None)

    def number(field, minimum=0.0):
        if field not in data:
            raise ValueError(f"'{field}' is required.", field)
        v = data[field]
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(f"'{field}' must be a number.", field)
        try:
            v = float(v)
        except ValueError:
            raise ValueError(f"'{field}' must be a number.", field) from None
        if not math.isfinite(v) or v < minimum or v > MAX_VALUE:
            raise ValueError(f"'{field}' must be between {minimum:g} and {MAX_VALUE:.0e}.", field)
        return v

    tx_type = data.get("type")
    if tx_type not in TYPES:
        raise ValueError(f"'type' must be one of {', '.join(TYPES)}.", "type")
    hour = number("hour")
    if hour != int(hour) or not 0 <= hour <= 23:
        raise ValueError("'hour' must be a whole number from 0 to 23.", "hour")
    return {
        "type": tx_type,
        "amount": number("amount"),
        "oldbalanceOrg": number("oldbalanceOrg"),
        "oldbalanceDest": number("oldbalanceDest"),
        "hour": int(hour),
    }


def create_app(model_path=None, metrics_path=None):
    model_path = Path(model_path or os.getenv("MODEL_PATH", BASE / "model_v2.pkl"))
    metrics_path = Path(metrics_path or os.getenv("METRICS_PATH", BASE / "metrics.json"))
    if not model_path.exists():
        raise RuntimeError(f"{model_path.name} not found. Run: python train_eval.py --csv Fraud.csv")

    # Only load model files you trained yourself: pickle can execute arbitrary code.
    bundle = joblib.load(model_path)
    pre = bundle["pipeline"].named_steps["pre"]
    clf = bundle["pipeline"].named_steps["clf"]
    threshold = float(bundle["threshold"])
    names = list(pre.named_steps["prep"].get_feature_names_out())

    app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")

    @app.get("/")
    def home():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", threshold=threshold)

    @app.get("/api/metrics")
    def metrics():
        if not metrics_path.exists():
            return jsonify(available=False)
        return jsonify(available=True, **json.loads(metrics_path.read_text()))

    @app.post("/api/predict")
    def predict():
        try:
            tx = parse_payload(request.get_json(silent=True))
        except ValueError as exc:
            message, field = exc.args
            return jsonify(error=message, field=field), 400

        X = pre.transform(pd.DataFrame([tx]))
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        score = float(clf.predict_proba(X)[0, 1])

        contribs = clf.get_booster().predict(xgb.DMatrix(X.astype(np.float32)), pred_contribs=True)[0][:-1]
        top = sorted(zip(names, contribs), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        reasons = [{"label": pretty(n), "impact": round(float(c), 3)} for n, c in top if abs(c) > 1e-6]

        return jsonify(flagged=score >= threshold, score=round(score, 6),
                       threshold=round(threshold, 6), reasons=reasons)

    return app


# Module-level `app` so a WSGI server (gunicorn, etc.) can import it as "app:app".
# create_app() requires model_v2.pkl to exist -- run train_eval.py before deploying.
app = create_app()

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
