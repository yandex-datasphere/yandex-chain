from yandex_chain import YandexLLM, YandexEmbeddings, __version__
import langchain
import time 

print(f"Using yandex_chain version {__version__}, langchain=={langchain.__version__}")
gpt = YandexLLM(config='tests/config.json')

id = gpt.invokeAsync('Привет! Придумай 10 новых слов для приветствия.')
print(f"Submitted operation id={id}")

print(f"Checking operation")
while True:
    res = gpt.checkAsyncResult(id)
    print(res)
    if res:
        break
    time.sleep(5)

print(f"Usage: {gpt.totalTokens} tokens")
