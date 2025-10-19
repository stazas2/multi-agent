FROM node:20-slim AS frontend-builder

WORKDIR /build

COPY web/package.json web/package-lock.json ./web/
WORKDIR /build/web
RUN npm ci
COPY web/ .
RUN npm run build

FROM python:3.11-slim AS runtime

WORKDIR /app

# Install orchestrator dependencies
COPY orchestrator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy orchestrator code and shared utilities
COPY orchestrator/ /app/
COPY shared/ /app/shared/

# Optional: include SDK if orchestrator imports it in the future
COPY sdk/ /app/sdk/

# Include compiled frontend bundle
COPY --from=frontend-builder /build/web/dist /app/frontend

# Expose environment settings expected by Cloud Run
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Start the orchestrator
CMD ["python", "main.py"]
