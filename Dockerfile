# PLYXIS console and API, on-prem or in a customer's tenant. No external services are needed at run time.
#   docker build -t plyxis . && docker run -p 8000:8000 -v plyxis-runs:/app/runs plyxis
# Not yet built or run: the machine this was written on has no Docker. Expect to adjust it on first use.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npx vite build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY plyxis/ plyxis/
COPY scenarios/ scenarios/
COPY bench/ bench/
COPY --from=web /web/dist web/dist
RUN python -m plyxis.native build || echo "native kernels not built: NumPy fallback"
EXPOSE 8000
CMD ["python", "-m", "plyxis", "serve", "--host", "0.0.0.0", "--port", "8000"]
