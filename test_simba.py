
import os
import numpy as np
from PIL import Image

ENV_MODEL = 'keras'
# ENV_MODEL = 'deepapi'

os.environ['ENV_MODEL'] = ENV_MODEL

from simba_attack import SimBA
from imagenet import load_imagenet

N_SAMPLES = 10

if __name__ == "__main__":

    x_test, y_test = load_imagenet(N_SAMPLES)

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

    # SimBA Attack
    simba = SimBA(model)
    x_adv = simba.attack(x_test, y_test, epsilon=0.05, max_it=1000)

    # Distributed SimBA Attack
    # x_adv = simba.attack(x, epsilon=0.1, max_it=1000, distributed=True , batch=50, max_workers=10)

    # Save the adversarial images
    for i, xa in enumerate(x_adv):
        if ENV_MODEL == 'keras':
            im = Image.fromarray(np.array(np.uint8(x_test[i])))
            im_adv = Image.fromarray(np.array(np.uint8(xa)))
        elif ENV_MODEL == 'deepapi':
            im = Image.fromarray(np.array(np.uint8(x_test[i]*255.0)))
            im_adv = Image.fromarray(np.array(np.uint8(xa*255.0)))

        im.save(f"images/x_{i}.jpg", quality=95)
        im_adv.save(f"images/x_{i}_adv.jpg", quality=95)
