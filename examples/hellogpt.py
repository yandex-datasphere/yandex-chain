from yandex_chain import YandexLLM, YandexEmbeddings

gpt = YandexLLM(config='tests/config.json')
print(gpt('Привет! Придумай 10 новых слов для приветствия.'))
print(f"Usage: {gpt.totalTokens} tokens")

emb = YandexEmbeddings(config='tests/config.json')
print(emb.embed_document('Hello, world'))
