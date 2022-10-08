import io
from google.cloud import vision

import concurrent.futures

class CloudVision:
    def __init__(self, concurrency=1):
        self.concurrency = concurrency

    def predict(self, image_path):
        client = vision.ImageAnnotatorClient()
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        labels = response.label_annotations

        return labels

    def predictX(self, image_paths):
        y_preds = []
        y_index = []
        y_executors = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, image_path in enumerate(image_paths):
                # Load the input image and construct the payload for the request
                y_executors[executor.submit(self.predict, image_path)] = i

            for y_executor in concurrent.futures.as_completed(y_executors):
                y_index.append(y_executors[y_executor])
                y_preds.append(y_executor.result())

            y_preds = [y for _, y in sorted(zip(y_index, y_preds))]

        return y_preds
