# Streamlit user-interface image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    API_URL=http://api:8000

WORKDIR /app

COPY requirements-frontend.txt ./requirements-frontend.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-frontend.txt

COPY --chown=10001:10001 frontend/streamlit_app.py ./frontend/streamlit_app.py
COPY --chown=10001:10001 .streamlit/config.toml ./.streamlit/config.toml

USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()"

CMD ["python", "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
