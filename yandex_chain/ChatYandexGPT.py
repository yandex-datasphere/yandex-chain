from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from typing import Any, Dict, List, Optional, Tuple, cast
from langchain.callbacks.manager import CallbackManagerForLLMRun
from .YandexGPT import YandexLLM
from .util import YException

class ChatYandexGPT(YandexLLM,BaseChatModel):

    def conv_message(self,msg : BaseMessage):
        if isinstance(msg, HumanMessage):
            return YandexLLM.UserMessage(msg.content)
        if isinstance(msg, AIMessage):
            return YandexLLM.AssistantMessage(msg.content)
        if isinstance(msg, SystemMessage):
            return YandexLLM.SystemMessage(msg.content)
        raise YException("Unknown message type")

    def __call__(self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any):
        return self._generate(messages, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any):
        msg = [self.conv_message(x) for x in messages]
        res = super()._generate_messages(msg, **kwargs)
        return AIMessage(content=res)
