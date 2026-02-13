FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /code

COPY requirements.txt .
# Wichtig: torch/torchvision/torchaudio NICHT nochmal installieren, sonst überschreibst du ggf. das Base-Setup
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "pretrain_mim_wandb.py"]
