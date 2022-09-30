r"""
This module implements the DeepAPI client for pretrained VGG16 model on CIFAR-10 dataset.
"""

import requests
import urllib.request, json 

import numpy as np
np.set_printoptions(suppress=True)

from PIL import Image
from io import BytesIO
import base64

import concurrent.futures

class VGG16ImageNet:
    def __init__(self, url):
        """
        - url: DeepAPI server URL
        """
        self.url = url

        # # Load the Keras application labels 
        with urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json") as l_url:
            imagenet_json = json.load(l_url)

            # imagenet_json['134'] = ["n02012849", "crane"],
            imagenet_json['517'] = ['n02012849', 'crane_bird']

            # imagenet_json['638'] = ['n03710637', 'maillot']
            imagenet_json['639'] = ['n03710721', 'maillot_tank_suit']

            self.imagenet_labels = [imagenet_json[str(k)][1] for k in range(len(imagenet_json))]

    def predictX(self, X, max_workers=4):
        """
        - X: numpy array of shape (N, H, W, C)
        """
        if isinstance(X, list):
            for x in X:
                assert len(x.shape) == 3, 'Expecting a 3D tensor'
        else:
            if len(X.shape) != 4 or X.shape[3] != 3:
                raise ValueError(
                    "`predict` expects "
                    "a batch of images "
                    "(i.e. a 4D array of shape (samples, 224, 224, 3)). "
                    "Found array with shape: " + str(X.shape)
                )

        # Single thread
        def send_request(url, data):
            y_pred_temp = np.zeros(len(self.imagenet_labels))
            res = requests.post(url, json=data).json()['predictions']
            for r in res:
                y_pred_temp[self.imagenet_labels.index(r['label'])] = r['probability']

            return y_pred_temp

        y_preds = []
        y_requests = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for x in X:
                # Load the input image and construct the payload for the request
                image = Image.fromarray(np.uint8(x * 255.0))
                buff = BytesIO()
                image.save(buff, format="JPEG")

                data = {'file': base64.b64encode(buff.getvalue()).decode("utf-8")}
                y_requests.append(executor.submit(send_request, url=self.url, data=data))

            for y_request in concurrent.futures.as_completed(y_requests):
                y_preds.append(y_request.result())

        return np.array(y_preds)


    def predict(self, X):
        """
        - X: numpy array of shape (N, H, W, C)
        """
        if isinstance(X, list):
            for x in X:
                assert len(x.shape) == 3, 'Expecting a 3D tensor'
        else:
            if len(X.shape) != 4 or X.shape[3] != 3:
                raise ValueError(
                    "`predict` expects "
                    "a batch of images "
                    "(i.e. a 4D array of shape (samples, 224, 224, 3)). "
                    "Found array with shape: " + str(X.shape)
                )

        y_preds = []
        try:
            y_pred_temp = np.zeros([len(self.imagenet_labels)])
            for x in X:
                # Load the input image and construct the payload for the request
                image = Image.fromarray(np.uint8(x * 255.0))
                buff = BytesIO()
                image.save(buff, format="JPEG")

                data = {'file': base64.b64encode(buff.getvalue()).decode("utf-8")}
                res = requests.post(self.url, json=data).json()['predictions']

                for r in res:
                    y_pred_temp[self.imagenet_labels.index(r['label'])] = r['probability']

                y_preds.append(y_pred_temp)

        except Exception as e:
            print(e)

        return np.array(y_preds)

    def print(self, y):
        """
        Print the prediction result.
        """
        print()
        for i in range(0, len(y)):
            print('{:<25s}{:.5f}'.format(self.imagenet_labels[i], y[i]))

    def get_class_name(self, i):
        """
        Get the class name from the prediction label 0-10.
        """
        return self.imagenet_labels[i]
