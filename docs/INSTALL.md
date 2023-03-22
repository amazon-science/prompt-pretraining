# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n pomp python=3.8

# Activate the environment
conda activate pomp

# Install torch and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Clone POMP code repository and install requirements
```bash
# Clone clip-openness code base
git clone https://github.com/amazon-science/pomp.git

cd pomp/

# Install requirements
pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```

* Install [dassl library](https://github.com/KaiyangZhou/Dassl.pytorch).
```bash
cd third_party/Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ../..
```

To apply the pre-trained POMP prompt to semantic segmentation and object detection, please refer to [ZSSeg](../third_party/zsseg.baseline/README.md) and [Detic](../third_party/Detic/README.md) codebases, respectively. 