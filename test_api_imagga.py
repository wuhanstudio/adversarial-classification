import time
import fiftyone.zoo as foz
from apis.imagga import Imagga

N_SAMPLES = 10

imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")

api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'

if __name__ == '__main__':

    image_paths = []
    imagga_client = Imagga(api_key, api_secret, concurrency=2)

    for sample in imagenet_dataset:
        image_paths.append(str(sample['filepath']))

    print('Running Imagga API')

    one_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        imagga_client.predict(image_paths[0])
        end = time.time()

        one_query.append(end - start)

    avg_one_query = sum(one_query) / len(one_query) * 1000
    print('Imagga API (1): {:.2f} ms'.format(avg_one_query))

    two_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        imagga_client.predictX(image_paths[:2])
        end = time.time()

        two_query.append(end - start)

    avg_five_query = sum(two_query) / len(two_query) * 1000
    print('Imagga API (2): {:.2f} ms, x{:.2f} faster'.format(avg_five_query, avg_one_query * 2 / avg_five_query))

    ten_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        imagga_client.predictX(image_paths[:10])
        end = time.time()

        ten_query.append(end - start)

    avg_ten_query = sum(ten_query) / len(ten_query) * 1000
    print('Imagga API (10): {:.2f} ms, x{:.2f} faster'.format(avg_ten_query, avg_one_query * 10 / avg_ten_query))

    twenty_query = []
    for i in range(N_SAMPLES):
        start = time.time()
        imagga_client.predictX(image_paths[:20])
        end = time.time()

        twenty_query.append(end - start)

    avg_twenty_query = sum(twenty_query) / len(twenty_query) * 1000
    print('Imagga API (20): {:.2f} ms, x{:.2f} faster'.format(avg_twenty_query, avg_one_query * 20 / avg_twenty_query))
