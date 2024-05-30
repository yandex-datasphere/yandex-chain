import requests
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain
from langchain_core.language_models.llms import LLM
from yandex_chain.util import YAuth, YException
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed
from enum import Enum

class YandexGPTModel(Enum):
    Lite = 0
    Pro = 1
    Summarization = 2
    LiteRC = 3
    Custom = 99

class YandexLLM(LLM):
    temperature : float = 1.0
    max_tokens : int = 1500
    sleep_interval : float = 1.0
    retries = 3
    use_lite : bool = None
    model : YandexGPTModel = None
    instruction_text : str = None
    instruction_id : str = None
    iam_token : str = None
    folder_id : str = None
    api_key : str = None
    config : str = None

    inputTextTokens : int = 0
    completionTokens : int = 0
    totalTokens : int = 0
    disable_logging : bool = False

    @property
    def _llm_type(self) -> str:
        return "YandexGPT"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return { "max_tokens": self.max_tokens, "temperature" : self.temperature }

    @property
    def _modelUri(self):
        if self.model is None:
            if self.use_lite is None:
                self.model = YandexGPTModel.Lite
            else:
                self.model = YandexGPTModel.Lite if self.use_lite else YandexGPTModel.Pro
        if self.instruction_id:
            self.model = YandexGPTModel.Custom
            return f"ds://{self.instruction_id}"
        if self.model == YandexGPTModel.Lite:
            return f"gpt://{self.folder_id}/yandexgpt-lite/latest"
        if self.model == YandexGPTModel.LiteRC:
            return f"gpt://{self.folder_id}/yandexgpt-lite/rc"
        elif self.model == YandexGPTModel.Pro:
            return f"gpt://{self.folder_id}/yandexgpt/latest"
        elif self.model == YandexGPTModel.Summarization:
            return f"gpt://{self.folder_id}/summarization/latest"
        else:
            raise YException("Invalid model flag")

    @staticmethod
    def UserMessage(message):
        return { "role" : "user", "text" : message }

    @staticmethod
    def AssistantMessage(message):
        return { "role" : "assistant", "text" : message }

    @staticmethod
    def SystemMessage(message):
        return { "role" : "system", "text" : message }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        return_message = False
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        msgs = []
        if self.instruction_text:
            msgs.append(self.SystemMessage(self.instruction_text))
        msgs.append(self.UserMessage(prompt))
        return self._generate_messages(msgs)

    def _generate_messages(self,
        messages : List[Mapping],       
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        return_message = False):
        auth = YAuth.from_params(dict(self))
        if not self.folder_id:
            self.folder_id = auth.folder_id
        req = {
          "modelUri": self._modelUri,
          "completionOptions": {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
          },
          "messages" : messages
        }
        headers = auth.headers
        if self.disable_logging:
            headers['x-data-logging-enabled'] = 'false'
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries),wait=wait_fixed(self.sleep_interval)):
                with attempt:
                    res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                            headers=headers, json=req)
                    js = res.json()
                    if 'result' in js:
                        res = js['result']
                        usage = res['usage']
                        self.totalTokens += int(usage['totalTokens'])
                        self.completionTokens += int(usage['completionTokens'])
                        self.inputTextTokens += int(usage['inputTextTokens'])
                        return res['alternatives'][0]['message'] if return_message else res['alternatives'][0]['message']['text']
                    raise YException(f"Cannot process YandexGPT request, result received: {js}")
        except RetryError:
            raise YException(f"Error calling YandexGPT after {self.retries} retries. Result returned:\n{js}")

    def resetUsage(self):
        self.totalTokens = 0
        self.completionTokens = 0
        self.inputTextTokens = 0
