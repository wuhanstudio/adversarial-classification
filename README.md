## Adversarial Classification

> Distributed Black-Box attacks against Image Classification.

[[ Talk ]](https://distributed.wuhanstudio.uk) [[ Video ]](https://youtu.be/3OTjkBSGhRw) [[ Paper ]](https://arxiv.org/abs/2210.16371) [[ Code ]](https://github.com/wuhanstudio/adversarial-classification)

Whether black-box attacks have become a real threat against image classification cloud services? Our research demonstrates that it is possible to exploit the load balancer to accelerate black-box attacks.

We open-source our image classification cloud service, [DeepAPI](https://github.com/wuhanstudio/deepapi), to facilitate future research on distributed blackbox attacks that bring adversarial attacks closer to being a practical threat against machine learning models deployed on cloud servers.

![](docs/distribution.jpg)


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

#### Test Cloud APIs

```
# Set up your Google Cloud Vision
$ python test_api_cloud_vision.py

# Change your API key in the code
$ python test_api_imagga.py

$ python test_api_deepapi.py
```

#### Horizontal Distribution

```
# You can change the target model and DeepAPI URL in the code.

# Bandits Attack
$ python test_attack_bandits.py

# SimBA Attack
$ python test_attack_simba.py

# Square Attack
$ python test_attack_square.py
```

#### Vertical Distribution

```
# You can change the target model and DeepAPI URL in the code.

# Bandits Attack
$ python test_attack_bandits_one.py

# SimBA Attack
$ python test_attack_simba_one.py

# Square Attack
$ python test_attack_square_one.py
```

# Black-box Adversarial Toolbox (BAT)

[![Build Status](https://app.travis-ci.com/wuhanstudio/blackbox-adversarial-toolbox.svg?branch=master)](https://app.travis-ci.com/wuhanstudio/blackbox-adversarial-toolbox)
[![PyPI version](https://badge.fury.io/py/blackbox-adversarial-toolbox.svg)](https://badge.fury.io/py/blackbox-adversarial-toolbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/blackbox-adversarial-toolbox)](https://pypi.org/project/blackbox-adversarial-toolbox/)
[![](https://img.shields.io/badge/Documentation-brightgreen)](https://bat.wuhanstudio.uk/)

Alternatively, you can try distributed black-box attacks using our toolbox.

## Installation

```python
pip install blackbox-adversarial-toolbox
```

<a href="https://github.com/wuhanstudio/blackbox-adversarial-toolbox"><img src="https://bat.wuhanstudio.uk/images/bat.png" width=300px style="float: left;" ></a>

## Usage (CLI)

```
Usage: bat [OPTIONS] COMMAND [ARGS]...

  The CLI tool for Black-box Adversarial Toolbox (BAT).

Options:
  --help  Show this message and exit.

Commands:
  api      Manage Cloud APIs
  attack   Manage Attacks
  example  Manage Examples
```
