FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN rm -f /etc/apt/sources.list.d/*.list
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
WORKDIR /app
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3.10-dev python3.10-venv python3-pip -y --no-install-recommends && \
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
RUN git clone https://github.com/PKU-YuanGroup/MoE-LLaVA
WORKDIR /app/MoE-LLaVA
RUN pip install -e .
RUN pip install -e ".[train]"
RUN pip install flash-attn --no-build-isolation
WORKDIR /app
ENV MODEL="LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384"
ENV HF_HOME="/"
RUN python3 download_model.py
CMD ["python3", "./worker.py"]