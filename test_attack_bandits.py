
import os
import numpy as np
from PIL import Image

# ENV_MODEL = 'keras'
ENV_MODEL = 'deepapi'

DEEP_API_URL = 'http://localhost:8080'

ENV_MODEL_TYPE = 'inceptionv3'
# ENV_MODEL_TYPE = 'resnet50'
# ENV_MODEL_TYPE = 'vgg16'

os.environ['ENV_MODEL'] = ENV_MODEL
os.environ['ENV_MODEL_TYPE'] = ENV_MODEL_TYPE

from attacks.bandits_attack import BanditsAttack
from dataset.imagenet import load_imagenet

N_SAMPLES = 100
CONCURRENCY = 8

if __name__ == "__main__":

    x_test, y_test = load_imagenet(N_SAMPLES)

    if ENV_MODEL == 'keras':
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    # Initialize the Cloud API Model
    if ENV_MODEL == 'deepapi':

        if ENV_MODEL_TYPE == 'inceptionv3':
            from apis.deepapi import DeepAPI_Inceptionv3_ImageNet
            model = DeepAPI_Inceptionv3_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)
        elif ENV_MODEL_TYPE == 'resnet50':
            from apis.deepapi import DeepAPI_Resnet50_ImageNet
            model = DeepAPI_Resnet50_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)
        elif ENV_MODEL_TYPE == 'vgg16':
            from apis.deepapi import DeepAPI_VGG16_ImageNet
            model = DeepAPI_VGG16_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)

    elif ENV_MODEL == 'keras':

        if ENV_MODEL_TYPE == 'inceptionv3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3
            model = InceptionV3(weights='imagenet')
        elif ENV_MODEL_TYPE == 'resnet50':
            from tensorflow.keras.applications.resnet50 import ResNet50
            model = ResNet50(weights='imagenet')
        elif ENV_MODEL_TYPE == 'vgg16':
            from tensorflow.keras.applications.vgg16 import VGG16
            model = VGG16(weights='imagenet')

    # Bandits Attack
    bandits_attack = BanditsAttack(model)

    # Horizontally Distributed Attack
    x_adv = bandits_attack.attack(x_test, y_test, epsilon = 0.05, max_it=1000, online_lr=100)

    # Vertically Distributed Attack
    # x_adv = []
    # for xt, yt in zip(x_test, y_test):
    #     bandits_attack.attack(x_test, y_test, epsilon = 0.05, max_it=1000, online_lr=100, concurrency=CONCURRENCY)
    #     x_adv.append(xa)

    for i, xa in enumerate(x_adv):
        im = Image.fromarray(np.array(np.uint8(x_test[i])))
        im_adv = Image.fromarray(np.array(np.uint8(xa)))

        im.save(f"images/x_{i}.jpg", subsampling=0, quality=100)
        im_adv.save(f"images/x_{i}_adv.jpg", subsampling=0, quality=100)
