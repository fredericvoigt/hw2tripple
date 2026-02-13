FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

# 1) toolchain in conda (funktioniert sicher in pytorch/pytorch images)
RUN conda install -y -c conda-forge \
    compilers \
    && conda clean -a -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "pretrain_mim_wandb.py"]
