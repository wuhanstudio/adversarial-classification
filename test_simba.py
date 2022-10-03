import numpy as np
from PIL import Image

from simba import SimBA
from deepapi import DeepAPI_VGG16Cifar10

# Load Image [0.0, 1.0]
x = np.asarray(Image.open('test/dog.jpg').resize((32, 32))) / 255.0

# Initialize the Cloud API Model
DEEP_API_URL = 'http://localhost:8080'
model = DeepAPI_VGG16Cifar10(DEEP_API_URL)

# SimBA Attack
simba = SimBA(model)
x_adv = simba.attack(x, epsilon=0.1, max_it=1000)

# Distributed SimBA Attack
x_adv = simba.attack(x, epsilon=0.1, max_it=1000, distributed=True , batch=50, max_workers=10)
