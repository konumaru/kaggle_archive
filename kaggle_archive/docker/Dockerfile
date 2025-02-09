FROM gcr.io/kaggle-gpu-images/python:latest

WORKDIR /workspace

RUN apt update
RUN apt install -y \
    git \
    bash-completion \
    python3-llvmlite \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    clinfo \ 
    opencl-headers \
    libboost-all-dev \
    ocl-icd-opencl-dev 

RUN echo "source /etc/bash_completion" >> /root/.bashrc

RUN pip install -U pip
RUN pip install -U black isort mypy pyproject-flake8
RUN pip install -U pytest-cov
RUN pip install -U iterative-stratification hydra-core
# RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge


RUN git config --global user.email "konumaru1022@gmail.com"
RUN git config --global user.name "konumaru"
RUN git config --global --add safe.directory /workspace
