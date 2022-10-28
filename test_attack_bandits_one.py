
import os
import numpy as np
from PIL import Image

from datetime import datetime

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
CONCURRENCY = 1

if __name__ == "__main__":

    x_test, y_test = load_imagenet(N_SAMPLES)

    if ENV_MODEL_TYPE == 'inceptionv3':
        from apis.deepapi import DeepAPI_Inceptionv3_ImageNet
        model = DeepAPI_Inceptionv3_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)
    elif ENV_MODEL_TYPE == 'resnet50':
        from apis.deepapi import DeepAPI_Resnet50_ImageNet
        model = DeepAPI_Resnet50_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)
    elif ENV_MODEL_TYPE == 'vgg16':
        from apis.deepapi import DeepAPI_VGG16_ImageNet
        model = DeepAPI_VGG16_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)

    # Bandits Attack
    bandits_attack = BanditsAttack(model)

    # Vertically Distributed Attack
    i = 29
    log_dir = 'logs/bandits/' + ENV_MODEL + '/one/' + ENV_MODEL_TYPE + '/' + str(i) + '/' + str(CONCURRENCY) + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    x_adv = bandits_attack.attack(np.array([x_test[i]]), np.array([y_test[i]]), epsilon = 0.05, max_it=1000, online_lr=100, concurrency=CONCURRENCY, log_dir=log_dir)

    im = Image.fromarray(np.array(np.uint8(x_test[i])))
    im_adv = Image.fromarray(np.array(np.uint8(x_adv[0])))

    im.save(f"x_{i}.jpg", subsampling=0, quality=100)
    im_adv.save(f"x_{i}_adv_bandits_" + str(CONCURRENCY) + ".jpg", subsampling=0, quality=100)
