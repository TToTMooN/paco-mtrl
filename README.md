# PaCo: Parameter-Compositional Multi-Task Reinforcement Learning (NeurIPS 2022)


## Introduction
This repo is for the Parameter-Compositional MTRL method proposed in ["PaCo: Parameter-Compositional Multi-Task Reinforcement Learning"](https://openreview.net/forum?id=LYXTPNWJLr).

## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
git clone --recursive https://github.com/TToTMooN/paco-mtrl.git
cd paco-mtrl

pip install -e . -e alf

```


### Training
```
cd paco/examples/
CUDA_VISIBLE_DEVICE=0 python3 -m alf.bin.train --conf=metaworld_mt10_paco_conf.py --root_dir=EXP_PATH
```
More examples will be added soon.

## Contact

For questions related to PaCo, please send me an email: ```lingfengsun@berkeley.edu```

