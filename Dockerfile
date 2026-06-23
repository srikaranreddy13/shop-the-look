# Streamlit on Hugging Face Spaces.
# HF deprecated the built-in Streamlit SDK, so Streamlit apps run via the Docker
# SDK. The README's `sdk: docker` + `app_port: 8501` tell the Space to build
# this file and expose port 8501.
FROM python:3.11-slim

# Run as a non-root user (HF Spaces best practice) with a writable home so the
# model weights can download to ~/.cache/huggingface at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8501

CMD ["streamlit", "run", "04_demo.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
