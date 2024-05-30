from yandex_chain import YandexLLM, YandexGPTModel

gpt_lite = YandexLLM(config='tests/config.json')
gpt_full = YandexLLM(config='tests/config.json',model = YandexGPTModel.Pro)

print("=== LITE MODEL ===")
print(f"URI: {gpt_lite._modelUri}")
print(gpt_lite('Привет! Придумай 10 новых креативных слов для приветствия, которыми могут пользоваться программисты.'))

print("=== FULL MODEL ===")
print(f"URI: {gpt_full._modelUri}")
print(gpt_full('Привет! Придумай 10 новых креативных слов для приветствия, которыми могут пользоваться программисты.'))

print("=== FULL BY PROPERTY CHANGE ===")
gpt_lite.model = YandexGPTModel.Pro
print(f"URI: {gpt_lite._modelUri}")
print(gpt_lite('Привет! Придумай 10 новых креативных слов для приветствия, которыми могут пользоваться программисты.'))
