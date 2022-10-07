import fiftyone.zoo as foz

from apis.google import CloudVision
from utils.timer import Timer

imagenet_dataset = foz.load_zoo_dataset("imagenet-sample")
imagenet_labels = foz.load_zoo_dataset_info("imagenet-sample").classes

if __name__ == '__main__':

    image_paths = []
    vision_client = CloudVision(concurrency=10)

    for sample in imagenet_dataset:
        image_paths.append(str(sample['filepath']))

    print('Running Cloud Vision API')

    with Timer('Cloud Vision Sequential'):
        res_1 = vision_client.predict(image_paths[0])

    with Timer('Cloud Vision Distributed - 5'):
        res_5 = vision_client.predictX(image_paths[:5])

    with Timer('Cloud Vision Distributed - 10'):
        res_10 = vision_client.predictX(image_paths[:10])

    print()
