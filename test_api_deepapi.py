import time

from dataset.imagenet import load_imagenet
from apis.deepapi import DeepAPI_VGG16_ImageNet

N_SAMPLES = 10

if __name__ == '__main__':

    x_test, y_test = load_imagenet(N_SAMPLES)

    image_paths = []
    DEEP_API_URL = 'https://deepapi.grayisland-ba9133aa.uksouth.azurecontainerapps.io/'
    deepapi_client = DeepAPI_VGG16_ImageNet(DEEP_API_URL, concurrency=8)

    print('Running Deep API')

    one_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        deepapi_client.predict(x_test[:1])
        end = time.time()

        one_query.append(end - start)

    avg_one_query = sum(one_query) / len(one_query) * 1000
    print('Deep API (1): {:.2f} ms'.format(avg_one_query))

    two_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        deepapi_client.predictX(x_test[:2])
        end = time.time()

        two_query.append(end - start)

    avg_five_query = sum(two_query) / len(two_query) * 1000
    print('Deep API (2): {:.2f} ms, x{:.2f} faster'.format(avg_five_query, avg_one_query * 2 / avg_five_query))

    ten_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        deepapi_client.predictX(x_test[:10])
        end = time.time()

        ten_query.append(end - start)

    avg_ten_query = sum(ten_query) / len(ten_query) * 1000
    print('Deep API (10): {:.2f} ms, x{:.2f} faster'.format(avg_ten_query, avg_one_query * 10 / avg_ten_query))

    twenty_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        deepapi_client.predictX(x_test[:20])
        end = time.time()

        twenty_query.append(end - start)

    avg_twenty_query = sum(twenty_query) / len(twenty_query) * 1000
    print('Deep API (20): {:.2f} ms, x{:.2f} faster'.format(avg_twenty_query, avg_one_query * 20 / avg_twenty_query))
