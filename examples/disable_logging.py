from yandex_chain import YandexLLM, YandexEmbeddings

gpt = YandexLLM(config='tests/config.json',disable_logging=True)
print(gpt('Привет! Придумай 10 новых слов для приветствия.'))
