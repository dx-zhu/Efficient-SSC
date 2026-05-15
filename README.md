# Efficient-SSC

# Get Started

## Environment building

```bash
conda create -n efficient python=3.8
conda activate efficient

pip install -r requirements.txt

cd src/third_party
pip install pointnet2_ops_lib/. # or git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib
cd ../..
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
apt-get install ninja-build
```

## Data preparation

### Semantic Scene Datasets

#### NYUCAD-PC

#### SSC-PC

[here](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md).