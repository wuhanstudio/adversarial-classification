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

from attacks.square_attack import SquareAttack
from dataset.imagenet import load_imagenet, imagenet_labels

N_SAMPLES = 100
CONCURRENCY = 8

def dense_to_onehot(y, n_classes):
    y_onehot = np.zeros([len(y), n_classes], dtype=bool)
    y_onehot[np.arange(len(y)), y] = True
    return y_onehot

if __name__ == '__main__':

    x_test, y_test = load_imagenet(N_SAMPLES)

    if ENV_MODEL == 'keras':
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    # Initialize the Cloud API Model
    if ENV_MODEL_TYPE == 'inceptionv3':
        from apis.deepapi import DeepAPI_Inceptionv3_ImageNet
        model = DeepAPI_Inceptionv3_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)
    elif ENV_MODEL_TYPE == 'resnet50':
        from apis.deepapi import DeepAPI_Resnet50_ImageNet
        model = DeepAPI_Resnet50_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)
    elif ENV_MODEL_TYPE == 'vgg16':
        from apis.deepapi import DeepAPI_VGG16_ImageNet
        model = DeepAPI_VGG16_ImageNet(DEEP_API_URL, concurrency=CONCURRENCY)

    y_target_onehot = dense_to_onehot(y_test, n_classes=len(imagenet_labels))

    # Note: we count the queries only across correctly classified images
    square_attack = SquareAttack(model)

    # Vertically Distributed Attack
    i = 29
    log_dir = 'logs/square/' + ENV_MODEL + '/one/' + ENV_MODEL_TYPE  + '/' + str(i) + '/' + str(CONCURRENCY) + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    x_adv, _ = square_attack.attack(np.array([x_test[i]]), np.array([y_target_onehot[i]]), False, epsilon = 0.05, max_it=1000, log_dir=log_dir, concurrency=CONCURRENCY)

    # Save the adversarial images
    im = Image.fromarray(np.array(np.uint8(x_test[i])))
    im_adv = Image.fromarray(np.array(np.uint8(x_adv[0])))

    im.save(f"x_{i}.jpg", subsampling=0, quality=100)
    im_adv.save(f"x_{i}_adv_square_" + str(CONCURRENCY) + ".jpg", subsampling=0, quality=100)
