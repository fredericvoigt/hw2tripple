FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .
# Beispiel: inference
CMD ["python", "mymain.py"]
