# Solr + FastAPI shim in one container for HF Spaces.
# Solr stays internal on 8983; FastAPI shim listens on 7860 (HF default).

FROM solr:10-slim

USER root

# Python + minimal build deps for fastapi/torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SOLR_HOME=/tmp/solr_home \
    HF_HOME=/tmp/hf_cache \
    HF_HUB_CACHE=/tmp/hf_cache/hub \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/sentence-transformers \
    SHIM_PORT=7860
WORKDIR /opt/app

# Pre-create writable cache dirs the solr user will use at runtime.
RUN mkdir -p /tmp/hf_cache/hub /tmp/hf_cache/sentence-transformers /tmp/solr_home && \
    chown -R solr:solr /tmp/hf_cache /tmp/solr_home

COPY requirements.txt /opt/app/requirements.txt
RUN python3 -m pip install --break-system-packages -r /opt/app/requirements.txt

# Pre-download the bge model into /tmp/hf_cache (now that HF_HOME is set above).
# Run as solr so file ownership matches the runtime user.
USER solr
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
                SentenceTransformer('BAAI/bge-small-en-v1.5')"
USER root

# App + entrypoint.
COPY app.py /opt/app/app.py
COPY entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

EXPOSE 7860

USER solr
ENTRYPOINT ["/opt/app/entrypoint.sh"]
