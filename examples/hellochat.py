from yandex_chain import ChatYandexGPT
from langchain.schema import AIMessage, HumanMessage, SystemMessage

gpt = ChatYandexGPT(config='tests/config.json')
print(gpt([HumanMessage(content='Привет! Придумай 10 новых слов для приветствия.')]))
print(f"Usage: {gpt.totalTokens} tokens")
