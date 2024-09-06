import unittest
from yandex_chain import YandexGPTClassifier, YException, YandexGPTModel

class TestYandexGPTClassifier(unittest.TestCase):

    descr = "Пожалуйста, определи, является ли предложение позитивным, негативным или нейтральным"
    labels = ["негативное","нейтральное","позитивное"]

    def test_classify(self):
        YCLS = YandexGPTClassifier(task_description = self.descr, labels=self.labels, config="tests/config.json")
        res = YCLS.invoke("Я ненавижу эти прогорклые французские булочки!")
        x = YCLS.get_top_label(res)
        self.assertEqual(len(res), 3)
        self.assertEqual(x, self.labels[0])

if __name__ == '__main__':
    unittest.main()