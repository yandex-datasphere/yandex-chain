from yandex_chain import YandexLLM, YandexEmbeddings, __version__
import langchain

print(f"Using yandex_chain version {__version__}, langchain=={langchain.__version__}")
gpt = YandexLLM(config='tests/config.json')
print(gpt.invoke('Привет! Придумай 10 новых слов для приветствия.'))
print(f"Usage: {gpt.totalTokens} tokens")

emb = YandexEmbeddings(config='tests/config.json')
print(emb.embed_document('Hello, world'))
