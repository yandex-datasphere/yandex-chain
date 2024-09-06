import time
import requests
from yandex_chain.util import YAuth, YException
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed
import json

class YandexGPTClassifier():

    def __init__(self, task_description, labels, samples=None, sleep_interval=1, retries=3, *args, **kwargs):
        self.sleep_interval = sleep_interval
        self.task_description = task_description
        self.labels = labels
        self.samples = samples
        self.args = args;
        self.kwargs = kwargs;
        self.auth = YAuth.from_params(kwargs)
        self.headers = self.auth.headers
        self.retries = retries
        
    def _getModelUri(self):
        return f"cls://{self.auth.folder_id}/yandexgpt/latest"

    def invoke(self, text):
        j = {
         "modelUri": self._getModelUri(),
         "taskDescription": self.task_description,
        "labels": self.labels,
        "text": text
        }
        if self.samples:
            j["samples"] = self.samples
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries),wait=wait_fixed(self.sleep_interval)):
                with attempt:
                    res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification",
                                        json=j,headers=self.headers)
                    js = res.json()
                    if 'predictions' in js.keys():
                        return js['predictions']
                    raise YException(f"No embedding found, result returned: {js}")
        except RetryError:
            raise YException(f"Error calling classifier after {self.retries} retries. Result returned:\n{js}")

    def get_top_label(self, result, return_confidence=False):
        x = sorted(result,key=lambda x:x['confidence'],reverse=True)[0]
        return x if return_confidence else x['label']