import unittest
from yandex_chain import YandexLLM, YException, YandexGPTModel

class TestYandexGPT(unittest.TestCase):

    def test_create_from_file(self):
        YGPT = YandexLLM(config="tests/config.json")
        res = YGPT.invoke('Imagine no possessions...')
        self.assertGreater(len(res), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_full_model(self):
        YGPT = YandexLLM(config="tests/config.json",use_lite=False)
        res = YGPT.invoke('Imagine no possessions...')
        self.assertGreater(len(res), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_literc_model(self):
        YGPT = YandexLLM(config="tests/config.json",model=YandexGPTModel.LiteRC)
        res  = YGPT.invoke('Imagine no possessions...')
        self.assertGreater(len(res), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_prorc_model(self):
        YGPT = YandexLLM(config="tests/config.json",model=YandexGPTModel.ProRC)
        res  = YGPT.invoke('Imagine no possessions...')
        self.assertGreater(len(res), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_async(self):
        YGPT = YandexLLM(config="tests/config.json")
        id = YGPT.invokeAsync('Imagine no possessions...')
        self.assertGreater(len(id), 3)

    def test_summarization_model(self):
        txt = 'Я как-то пошел гулять со своей девушкой, и увидел, что на дереве сидела птица воробей. Как дела? - спросил я у птицы воробья. Но птица ничего мне не отвечала, только сидела на дереве, потряхивая трухлявыми крыльями, и молчала.'
        YGPT = YandexLLM(config="tests/config.json",model=YandexGPTModel.Summarization)
        res = YGPT.invoke(txt)
        self.assertGreater(len(res), 1)
        # self.assertLess(len(res), len(txt))
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_options(self):
        YGPT = YandexLLM(config="tests/config.json",disable_logging=True)
        res = YGPT.invoke('Imagine no possessions...')
        self.assertGreater(len(res), 10)

    def wrong_auth(self):
        YGPT = YandexLLM(folder_id='xxxxxxx',iam_token='xxxxxxx')
        res = YGPT.invoke('Hello, world')

    def test_wrong_auth(self):
        self.assertRaises(YException,self.wrong_auth)

if __name__ == '__main__':
    unittest.main()