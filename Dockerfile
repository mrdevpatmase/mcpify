FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer is cached across builds that only
# change application code, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render still deploys via render.yaml's native Python env, not this
# Dockerfile - this exists so the service can be run/deployed anywhere
# else (AWS App Runner, Fargate, Cloud Run, local docker) without
# depending on Render-specific tooling. $PORT defaults to Render's own
# convention (10000) but is overridable, since other platforms
# (Cloud Run, App Runner) inject their own PORT value at runtime.
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
