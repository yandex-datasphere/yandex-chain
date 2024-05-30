from yandex_chain import YandexLLM, YandexGPTModel

gpt_sum = YandexLLM(config='tests/config.json',model=YandexGPTModel.Summarization)

txt = 'Я как-то пошел гулять со своей девушкой, и увидел, что на дереве сидела птица воробей. Как дела? - спросил я у птицы воробья. Но птица ничего мне не отвечала, только сидела на дереве, потряхивая трухлявыми крыльями, и молчала.'

print(f"Original text:\n{txt}\n-------\n")
s = gpt_sum.invoke(txt)

print(f"Summarized text:\n{s}\n-------\n")
