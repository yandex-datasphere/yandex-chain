from yandex_chain import YandexGPTClassifier, __version__

descr = "Пожалуйста, определи, является ли предложение позитивным, негативным или нейтральным"
classes = ["негативное","нейтральное","позитивное"]

sents = [
    "Я ненавижу тебя! Ты мерзкий тип!",
    "Какой однако сегодня прекрасный день!",
    "Сколько времени?"
]

samples = [
    {
        "text": "Ну какого чёрта ты не пожарил булочек на обед?",
        "label": "негативное"
    },
    {
        "text": "Ты молодец, булочки на обед сегодня были очень вкусными!",
        "label" : "позитивное"
    },
    {
        "text": "Сегодня новости ничего не говорят о свежих французских булках",
        "label" : "нейтральное"
    }
]

print(f"Using yandex_chain version {__version__}")

print('Zero-shot')
classifier = YandexGPTClassifier(task_description=descr, labels=classes, config='tests/config.json')
for s in sents:
    res = classifier.invoke(s)
    print(f" + [{classifier.get_top_label(res)}] {s}")

print('Few-shot')
classifier = YandexGPTClassifier(task_description=descr, labels=classes, config='tests/config.json',samples=samples)
for s in sents:
    res = classifier.invoke(s)
    print(f" + [{classifier.get_top_label(res)}] {s}")
