from yandex_chain import YandexLLM

uri = "ds://bt1l7p91d6vcnen0uss4"
uri = "ds://bt1954ej08dkghfetgro"
model = YandexLLM(config="tests/config.json",instruction_uri=uri)

res = model('Что такое цифровой рубль?')

print(res)