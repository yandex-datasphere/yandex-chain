from yandex_chain import YandexLLM

instruction_id = "bt1l7p91d6vcnen0uss4"
model = YandexLLM(config="tests/config.json",instruction_id=instruction_id)

res = model('Что такое цифровой рубль?')

print(res)