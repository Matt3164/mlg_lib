# mlg_lib

> Wrappers I used to try things 


This file will become your README and also the index of your documentation.

## Install

From pypy:

`pip install mlg_lib`

In dev mode:

```
git clone https://github.com/Matt3164/mlg_lib.git
cd mlg_lib
pip install -e .
```

## Main idea

The main idea here is to build image ML models with "simple" sklearn pipelines. Using simple ML pipelines to do machine learning on images is less and less done in profit of big DL models. 

In fact, I usually find handy to train a simple image feature model for at least two things : get a feel of the problem diificulty without the need of a GPU (I can often work on the problem modelisation before using a GPU...) and design the rest of the pipeline (data preprocessing, model deployment, GUI, debug statistics).

The main requisites for this lib were not have too much dependencies except classical numerical python.

It is still under heavy development, so API can change very quickly.


## How to use

TODO: Fill me in please! Don't forget code examples:

## TODOs

- [X] Fix daisy on fmnist
- [ ] Option to cache batches if no data augmentation
- [X] Compare to raw sklearn
- [ ] Add PCA Transform API
- [X] Rocket example
- [ ] Gabor features example
- [ ] Unsupervised features training ( PCA + Andrez Ng)
- [ ] Process to save data used for training : cleaner
- [ ] Implements FMIX