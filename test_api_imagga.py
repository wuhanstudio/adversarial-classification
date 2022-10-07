import fiftyone.zoo as foz

from apis.imagga import Imagga
from utils.timer import Timer

imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")
imagenet_labels = foz.load_zoo_dataset_info("imagenet-sample").classes

api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'

if __name__ == '__main__':

    image_paths = []
    imagga_client = Imagga(api_key, api_secret, concurrency=2)

    for sample in imagenet_dataset:
        image_paths.append(str(sample['filepath']))

    print('Running Imagga API')

    with Timer('Imagga Sequential'):
        for image_path in image_paths[:1]:
            imagga_client.predict(image_path)

    with Timer('Imagga Distributed - 5'):
        imagga_client.predictX(image_paths[:5])

    with Timer('Imagga Distributed - 10'):
        imagga_client.predictX(image_paths[:10])
