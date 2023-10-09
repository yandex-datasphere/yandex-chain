import requests
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain
from yandex_chain.util import YAuth, YException
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed

class YandexLLM(langchain.llms.base.LLM):
    temperature : float = 1.0
    max_tokens : int = 1500
    sleep_interval : float = 1.0
    retries = 3
    instruction_text : str = None
    iam_token : str = None
    folder_id : str = None
    api_key : str = None
    config : str = None

    @property
    def _llm_type(self) -> str:
        return "YandexGPT"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return { "max_tokens": self.max_tokens, "temperature" : self.temperature }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        auth = YAuth.from_params(dict(self))
        req = {
          "model": "general",
          "instruction_text": self.instruction_text,
          "request_text": prompt,
          "generation_options": {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
          }
        }
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries),wait=wait_fixed(self.sleep_interval)):
                with attempt:
                    res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/instruct",
                            headers=auth.headers, json=req)
                    js = res.json()
                    if 'result' in js:
                        return js['result']['alternatives'][0]['text']
                    raise Exception(f"Cannot process YaGPT request, result received: {js}")
        except RetryError:
            raise YException(f"Error calling YandexGPT after {self.retries} retries. Result returned:\n{js}")
