FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

# --- build essentials for packages with native extensions (evdev, etc.) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "pretrain_mim_wandb.py"]
