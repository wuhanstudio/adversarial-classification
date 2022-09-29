import time

import numpy as np
from PIL import Image

from io import BytesIO
import base64

import requests
import urllib.request, json 

import fiftyone.zoo as foz

imagenet_labels = []

N_SAMPLES = 100

def single_prediction(X, url):
    y_pred = []
    try:
        y_pred_temp = np.zeros([len(imagenet_labels)])
        for x in X:
            # Load the input image and construct the payload for the request
            image = Image.fromarray(np.uint8(x * 255.0))
            buff = BytesIO()
            image.save(buff, format="JPEG")

            data = {'file': base64.b64encode(buff.getvalue()).decode("utf-8")}
            res = requests.post(url, json=data).json()['predictions']

            for r in res:
                y_pred_temp[imagenet_labels.index(r['label'])] = r['probability']

            y_pred.append(y_pred_temp)

    except Exception as e:
        print(e)

    return np.array(y_pred)

if __name__ == '__main__':

    # Load the Keras application labels 
    with urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json") as l_url:
        imagenet_json = json.load(l_url)

        # imagenet_json['134'] = ["n02012849", "crane"],
        imagenet_json['517'] = ['n02012849', 'crane_bird']

        # imagenet_json['638'] = ['n03710637', 'maillot']
        imagenet_json['639'] = ['n03710721', 'maillot_tank_suit']

        imagenet_labels = [imagenet_json[str(k)][1] for k in range(len(imagenet_json))]

    # Load the dataset
    imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")

    X = []
    y = []
    for sample in imagenet_dataset:
        img = Image.open(str(sample['filepath']))

        X.append(np.array(img) / 255.0)
        y.append(sample['ground_truth']['label'])

    X = X[:N_SAMPLES]
    y = y[:N_SAMPLES]

    # Make predictions
    tm1 = time.perf_counter()

    y_preds = []
    for x in X:
        y_preds.append(single_prediction(np.array([x]), 'http://localhost:8080/vgg16'))

    tm2 = time.perf_counter()

    # Print the results
    for i, y_pred in enumerate(y_preds):
        print('Prediction', imagenet_labels[np.argmax(y_pred)], 'for', y[i])

    print(f'Total time elapsed: {tm2-tm1:0.2f} seconds')
