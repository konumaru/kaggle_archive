#! /bin/bash

# Install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

source ~/.bashrc

pyenv install anaconda3-2020.07
pyenv global anaconda3-2020.07

echo 'export PATH="$PYENV_ROOT/versions/anaconda3-2.5.0/bin/:$PATH"' >> ~/.bashrc

source ~/.bashrc

conda update -y conda
conda create -y --name rapids-0.16 \
     -c rapidsai \
     -c nvidia \
     -c conda-forge \
     -c defaults \
        rapids=0.16 \
        python=3.8 \
        cudatoolkit=11.0 \
        lightgbm \
        xgboost \
        matplotlib

source activate rapids-0.16
