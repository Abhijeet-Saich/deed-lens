# DeedLens: one image with Tesseract, CPU-only PyTorch, the app, the web page and the samples.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    DATA_DIR=/app/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: far smaller than the default CUDA build, and all DeedLens needs.
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Dependencies first (cached until pyproject.toml changes), then the code.
COPY pyproject.toml README.md LICENSE ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
      > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt
COPY deedlens ./deedlens
RUN pip install --no-deps .

COPY web ./web
COPY samples/*.pdf ./samples/
COPY eval ./eval
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
 && useradd --create-home --uid 1000 deedlens \
 && mkdir -p /app/data /models \
 && chown -R deedlens:deedlens /app /models

EXPOSE 8000
# Starts as root only to fix volume ownership, then drops to the deedlens user (see entrypoint).
ENTRYPOINT ["/entrypoint.sh"]
