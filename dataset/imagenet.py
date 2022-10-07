import os
import numpy as np
from PIL import Image
import fiftyone.zoo as foz

ENV_MODEL = os.environ.get('ENV_MODEL')

if ENV_MODEL is None:
    ENV_MODEL = 'deepapi'

def load_imagenet(n_samples):
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

    x_test = x_test[:n_samples]
    y_test = y_test[:n_samples]

    return x_test, y_test
