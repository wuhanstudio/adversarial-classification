import os
import numpy as np
from PIL import Image
import fiftyone.zoo as foz

ENV_MODEL = os.environ.get('ENV_MODEL')
ENV_MODEL_TYPE = os.environ.get('ENV_MODEL_TYPE')

if ENV_MODEL is None:
    ENV_MODEL = 'deepapi'

if ENV_MODEL_TYPE is None:
    ENV_MODEL_TYPE = 'inceptionv3'

imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")

imagenet_labels = foz.load_zoo_dataset_info("imagenet-sample").classes

def load_imagenet(n_samples):
    x_test = []
    y_test = []

    for sample in imagenet_dataset:
        x = Image.open(str(sample['filepath']))
        y = imagenet_labels.index(sample['ground_truth']['label'])

        if ENV_MODEL == 'keras':
            if ENV_MODEL_TYPE == 'inceptionv3':
                x = x.resize((299, 299))
            elif ENV_MODEL_TYPE == 'resnet50':
                x = x.resize((224, 224))
            elif ENV_MODEL_TYPE == 'vgg16':
                x = x.resize((224, 224))

        x = np.array(x)

        x_test.append(x)
        y_test.append(y)

    x_test = x_test[:n_samples]
    y_test = y_test[:n_samples]

    return x_test, y_test
