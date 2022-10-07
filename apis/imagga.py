import requests
import concurrent.futures

class Imagga:

    def __init__(self, api_key, api_secret, concurrency=1):
        self.api_key = api_key
        self.api_secret = api_secret
        self.concurrency = concurrency

        self.url = 'https://api.imagga.com/v2/tags'

    def predict(self, image_path):
        response = requests.post(
            self.url,
            auth=(self.api_key, self.api_secret),
            files={'image': open(image_path, 'rb')})

        return response.json()['result']['tags']

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
