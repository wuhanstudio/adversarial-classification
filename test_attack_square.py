from datetime import datetime

from apis.deepapi import DeepAPI_VGG16_ImageNet
import fiftyone.zoo as foz

import numpy as np
from PIL import Image

from attacks.square_attack import SquareAttack

def dense_to_onehot(y, n_classes):
    y_onehot = np.zeros([len(y), n_classes], dtype=bool)
    y_onehot[np.arange(len(y)), y] = True
    return y_onehot

N_SAMPLES = 100

if __name__ == '__main__':

    x_test = []
    y_test = []

    imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")
    imagenet_labels = foz.load_zoo_dataset_info("imagenet-sample").classes

    for sample in imagenet_dataset:
        x = Image.open(str(sample['filepath']))
        y = imagenet_labels.index(sample['ground_truth']['label'])

        x_test.append(np.array(x) / 255.0)
        y_test.append(y)

    x_test = x_test[:N_SAMPLES]
    y_test = y_test[:N_SAMPLES]

    model = DeepAPI_VGG16_ImageNet('http://localhost:8080')

    y_target_onehot = dense_to_onehot(y_test, n_classes=len(imagenet_labels))

    # Note: we count the queries only across correctly classified images
    square_attack = SquareAttack(model)
    log_dir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # x_adv, n_queries = square_attack.attack(x_test, y_target_onehot, False, n_iters=3000, distributed=False, log_dir=log_dir)
    x_adv, n_queries = square_attack.attack(x_test, y_target_onehot, False, n_iters=3000, distributed=True, batch=16, log_dir=log_dir)

    for i, xa in enumerate(x_adv):
        im = Image.fromarray(np.array(np.uint8(x_test[i]*255.0)))
        im_adv = Image.fromarray(np.array(np.uint8(xa*255.0)))
        im.save(f"images/x_{i}.jpg")
        im_adv.save(f"images/x_{i}_adv.jpg")
