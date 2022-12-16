# Towards Unifying Reference Expression Generation and Comprehension
This repository contains the code and model checkpoints for our paper ["Towards Unifying Reference Expression Generation and Comprehension"](https://arxiv.org/abs/2210.13076).

## Data Preparation
 To run the finetuning script, you should first download the original RefCOCO, RefCOCO+ and RefCOCOg from https://github.com/lichengunc/refer2, and put them at ``data/finetune``. Then you need to download the processed data from https://pan.baidu.com/s/1Pr83fXbyPI788CudSzSDiw (extraction code: s2gg), unzip and put it at ``data/finetune``.

## Pretrained Checkpoints
We release the pretrained checkpoints on RefCOCO/RefCOCO+/RefCOCOg at https://pan.baidu.com/s/1IWOMt1wvWzRNGC8XUSEs-w (extraction code: w4ux).

## Quick Usage
We provide interfaces at ``reg_inference.py`` and ``rec_inference.py``, which could help you to process your own data easily.

## Generated Results
The generated results of RefCOCO/RefCOCO+/RefCOCOg is at ``results/`` directory, including the corresponding BLEU, Meteor, Rouge, CIDEr values.

## Citation and Contact
If you find this work is useful or use the data in your work, please consider cite our paper:
```
@misc{https://doi.org/10.48550/arxiv.2210.13076,
  doi = {10.48550/ARXIV.2210.13076},
  url = {https://arxiv.org/abs/2210.13076},
  author = {Zheng, Duo and Kong, Tao and Jing, Ya and Wang, Jiaan and Wang, Xiaojie},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Towards Unifying Reference Expression Generation and Comprehension},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
If you have questions and suggestions, please feel free to email me (zd[at]bupt.edu.cn).
