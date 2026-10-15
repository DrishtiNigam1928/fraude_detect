# Containerizing and deploying FraudShield

Two parts: (1) build the model locally once, since Fraud.csv is too big to ship
in a container or commit to GitHub; (2) containerize and deploy only the
lightweight serving app.

## 0. Prerequisite — train once, locally

```powershell
python train_eval.py --csv Fraud.csv
```
This produces `model_v2.pkl` and `metrics.json` in the project folder. These
two small files ARE committed to git and ARE copied into the Docker image.
`Fraud.csv` itself is NOT committed (too large, and not needed to serve).

## 1. Test the container locally

Install Docker Desktop if you don't have it: https://www.docker.com/products/docker-desktop/

```powershell
docker build -t fraudshield .
docker run -p 5000:5000 fraudshield
```
Open http://127.0.0.1:5000 — same app, now running inside a container instead
of your venv. If this works, deployment will work the same way.

Common issue: if `docker build` fails saying `model_v2.pkl not found`, you
haven't run step 0 yet, or you're building from the wrong folder.

## 2. Push to GitHub

```powershell
git init
git add .
git commit -m "Fraud detection: leak-aware model + Docker deploy"
git branch -M main
git remote add origin https://github.com/<your-username>/fraud-detect.git
git push -u origin main
```
(Create the empty repo on github.com first, then paste its URL above.)

Double-check `model_v2.pkl` actually got committed (`git ls-files | findstr model_v2`) —
`.gitignore` excludes `Fraud.csv` but keeps the trained model.

## 3. Deploy — Render (free tier, easiest)

1. Go to https://render.com → sign in with GitHub.
2. **New +** → **Web Service** → pick your `fraud-detect` repo.
3. Render auto-detects the `Dockerfile`. Leave build/start command blank — the
   Dockerfile's `CMD` handles it.
4. Instance type: **Free**.
5. Click **Create Web Service**. First build takes a few minutes.
6. You get a URL like `https://fraud-detect-xxxx.onrender.com` — that's your
   live link. Put it in your resume and on the project's GitHub README.

Free tier note: Render's free web services sleep after 15 minutes of no
traffic and take ~30-60s to wake on the next request. Mention this if an
interviewer opens it cold — it's a hosting-tier limitation, not a bug.

## Alternatives to Render

- **Railway** (railway.app) — similar flow, also auto-detects Dockerfile, has
  a small free usage credit rather than a sleep policy.
- **Fly.io** — `fly launch` in the project folder detects the Dockerfile and
  deploys; free allowance covers one small always-on instance.

Any of these works the same way once the Dockerfile is in place — the
Dockerfile is the portable part, the platform is just where it runs.

## What NOT to do

- Don't try to run `train_eval.py` inside the Docker build — it needs
  `Fraud.csv` (huge) and `shap`/`matplotlib` (training-only deps this image
  deliberately excludes to stay small and fast to build).
- Don't commit `Fraud.csv` to git — GitHub will reject files over 100MB, and
  even under that it bloats the repo for no reason since the trained model is
  the only artifact serving actually needs.
