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
    python3-dev

RUN pip install -U pip
RUN pip install flake8 black isort autoflake
RUN pip3 uninstall -y torch torchvision
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install iterative-stratification

# Setting jupyter_notebook_config
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.notebook_dir = '/workspace'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

RUN git config --global user.email "konumaru1022@gmail.com"
RUN git config --global user.name "konumaru"
RUN git config --global --add safe.directory /workspace
