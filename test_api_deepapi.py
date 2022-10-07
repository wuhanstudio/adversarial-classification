from dataset.imagenet import load_imagenet
from apis.deepapi import DeepAPI_VGG16_ImageNet
from utils.timer import Timer

N_SAMPLES = 10

if __name__ == '__main__':

    x_test, y_test = load_imagenet(N_SAMPLES)

    image_paths = []
    DEEP_API_URL = 'http://localhost:8080'
    deepapi_client = DeepAPI_VGG16_ImageNet(DEEP_API_URL, concurrency=10)

    print('Running Deep API')

    with Timer('DeepAPI Sequential'):
        deepapi_client.predict(x_test[:1])

    with Timer('DeepAPI Distributed - 5'):
        deepapi_client.predictX(x_test[:5])

    with Timer('DeepAPI Distributed - 10'):
        deepapi_client.predictX(x_test[:10])
