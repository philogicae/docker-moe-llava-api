FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN rm -f /etc/apt/sources.list.d/*.list
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
WORKDIR /app
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libglib2.0-0 libx11-6 cmake libgtk2.0-0 libopenmpi-dev software-properties-common ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3.10-dev python3.10-venv python3-pip gcc g++ -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
COPY ./src/* ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r ./requirements.txt --no-cache-dir && \
    rm ./requirements.txt
RUN pip install -q torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 torchtext==0.16.0 torchdata==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121 -U
RUN pip install imageio-ffmpeg numpy==1.23.0 pandas pyngrok deepspeed==0.12.6 decord==0.6.0 transformers==4.37.0 einops timm tiktoken accelerate mpi4py
RUN git clone -b dev https://github.com/camenduru/MoE-LLaVA-hf /app/MoE-LLaVA-hf
WORKDIR /app/MoE-LLaVA-hf
RUN pip install -e .
WORKDIR /app
ENV MODEL="LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384"
ENV HF_HOME="/"
RUN python3 download_model.py
CMD ["python3", "./worker.py"]