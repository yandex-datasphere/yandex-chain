from yandex_chain import YandexLLM, YandexEmbeddings

gpt_lite = YandexLLM(config='tests/config.json')
gpt_full = YandexLLM(config='tests/config.json',use_lite=False)

print("=== LITE MODEL ===")
print(gpt_lite('Привет! Придумай 10 новых креативных слов для приветствия, которыми могут пользоваться программисты.'))

print("=== FULL MODEL ===")
print(gpt_full('Привет! Придумай 10 новых креативных слов для приветствия, которыми могут пользоваться программисты.'))
