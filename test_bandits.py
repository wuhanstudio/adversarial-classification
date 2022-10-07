
import os
import numpy as np
from PIL import Image
import fiftyone.zoo as foz

# Set environment variables

# ENV_MODEL = 'keras'
ENV_MODEL = 'deepapi'

os.environ['ENV_MODEL'] = ENV_MODEL

from bandits_attack import BanditsAttack

N_SAMPLES = 10

if __name__ == "__main__":

    x_test = []
    y_test = []

    imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")
    imagenet_labels = foz.load_zoo_dataset_info("imagenet-sample").classes

    for sample in imagenet_dataset:
        x = Image.open(str(sample['filepath']))
        y = imagenet_labels.index(sample['ground_truth']['label'])

        if ENV_MODEL == 'keras':
            x = x.resize((224, 224))
            x = np.array(x)

        if ENV_MODEL == 'deepapi':
            x = (np.array(x) / 255.0)

        x_test.append(x)
        y_test.append(y)

    x_test = x_test[:N_SAMPLES]
    y_test = y_test[:N_SAMPLES]

    if ENV_MODEL == 'keras':
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    # Initialize the Cloud API Model
    if ENV_MODEL == 'deepapi':
        from deepapi import DeepAPI_VGG16_ImageNet

        DEEP_API_URL = 'http://localhost:8080'
        model = DeepAPI_VGG16_ImageNet(DEEP_API_URL)

    elif ENV_MODEL == 'keras':
        from tensorflow.keras.applications.vgg16 import VGG16

        model = VGG16(weights='imagenet')

    # Bandits Attack
    bandits_attack = BanditsAttack(model)
    x_adv = bandits_attack.attack(x_test, y_test, max_it=10000)

    for i, xa in enumerate(x_adv):
        if ENV_MODEL == 'keras':
            im = Image.fromarray(np.array(np.uint8(x_test[i])))
            im_adv = Image.fromarray(np.array(np.uint8(xa)))
        elif ENV_MODEL == 'deepapi':
            im = Image.fromarray(np.array(np.uint8(x_test[i]*255.0)))
            im_adv = Image.fromarray(np.array(np.uint8(xa*255.0)))

        im.save(f"images/x_{i}.jpg", quality=95)
        im_adv.save(f"images/x_{i}_adv.jpg", quality=95)
