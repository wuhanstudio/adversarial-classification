## Adversarial Classification

> Distributed Black-Box attacks against Image Classification.

[[ Talk ]](https://distributed.wuhanstudio.uk) [[ Video ]]() [[ Paper ]](https://arxiv.org/abs/2210.16371) [[ Code ]](https://github.com/wuhanstudio/adversarial-classification)

Whether black-box attacks are real threats or just research stories?


### Quick Start

You may use [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). 

```python
$ git clone https://github.com/wuhanstudio/adversarial-classification
$ cd adversarial-classification

$ # CPU
$ conda env create -f environment.yml
$ conda activate adversarial-classification

$ # GPU
$ conda env create -f environment_gpu.yml
$ conda activate adversarial-gpu-classification
```

![](docs/distribution.jpg)

## Citation

```
@misc{han2022class,
  doi = {10.48550/ARXIV.2210.16371},
  url = {https://arxiv.org/abs/2210.16371},
  author = {Wu, Han and Rowlands, Sareh and Wahlstrom, Johan},
  title = {Distributed Black-box Attack against Image Classification Cloud Services},
  publisher = {arXiv},
  year = {2022}
}
```
