import unittest
from yandex_chain import YandexLLM, YException

class TestYandexGPT(unittest.TestCase):

    def test_create_from_file(self):
        YGPT = YandexLLM(config="tests/config.json")
        res = YGPT('Imagine no possessions...')
        self.assertGreater(len(res), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def test_full_model(self):
        YGPT = YandexLLM(config="tests/config.json",use_lite=False)
        res = YGPT('Imagine no possessions...')
        self.assertGreater(len(res), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def wrong_auth(self):
        YGPT = YandexLLM(folder_id='xxxxxxx',iam_token='xxxxxxx')
        res = YGPT('Hello, world')

    def test_wrong_auth(self):
        self.assertRaises(YException,self.wrong_auth)

if __name__ == '__main__':
    unittest.main()