FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# LightGBM / XGBoost (FLAML) precisam de libgomp no Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bootstrap.py txtai_store.py config_loader.py auth.py flowise_client.py lmstudio_client.py duckdb_store.py chat_store.py text_utils.py app.py ./
COPY ml_traditional ./ml_traditional
COPY sample_data ./sample_data
COPY store_docs ./store_docs
COPY .streamlit ./.streamlit
COPY config.toml.example ./config.toml.example

RUN cp config.toml.example config.toml \
    && mkdir -p /app/data /app/data/ml /app/data/ml/models

EXPOSE 8501

CMD ["sh", "-c", "python bootstrap.py; exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --browser.gatherUsageStats=false"]
