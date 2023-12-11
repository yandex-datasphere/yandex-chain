from langchain.embeddings.base import Embeddings
import time
import requests
import requests
from yandex_chain.util import YAuth, YException
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed

class YandexEmbeddings(Embeddings):

    def __init__(self, sleep_interval=1, retries=3, *args, **kwargs):
        self.sleep_interval = sleep_interval
        self.args = args;
        self.kwargs = kwargs;
        self.auth = YAuth.from_params(kwargs)
        self.headers = self.auth.headers
        self.retries = retries
        
    def _getModelUri(self,is_document=False):
        return f"emb://{self.auth.folder_id}/text-search-{'doc' if is_document else 'query'}/latest"

    def _embed(self, text, is_document=False):
        j = {
          "modelUri" : self._getModelUri(is_document),
          "text": text
        }
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries),wait=wait_fixed(self.sleep_interval)):
                with attempt:
                    res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
                                        json=j,headers=self.headers)
                    js = res.json()
                    if 'embedding' in js:
                        return js['embedding']
                    raise YException(f"No embedding found, result returned: {js}")
        except RetryError:
            raise YException(f"Error computing embeddings after {self.retries} retries. Result returned:\n{js}")

    def embed_document(self, text):
        return self._embed(text,is_document=True)

    def embed_documents(self, texts, chunk_size = 0):
        res = []
        for x in texts:
            res.append(self.embed_document(x))
            time.sleep(self.sleep_interval)
        return res
        
    def embed_query(self, text):
        return self._embed(text,is_document=False)
