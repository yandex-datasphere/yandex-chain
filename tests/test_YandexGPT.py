import unittest
import json
from yandex_chain import YandexLLM, YException

class TestYandexGPT(unittest.TestCase):

    def test_create_from_file(self):
        YGPT = YandexLLM(config="tests/config.json")
        res = YGPT('Imagine no possessions...')
        self.assertGreater(len(res), 10)

    def wrong_auth(self):
        YGPT = YandexLLM(folder_id='xxxxxxx',iam_token='xxxxxxx')
        res = YGPT('Hello, world')

    def test_wrong_auth(self):
        self.assertRaises(YException,self.wrong_auth)

if __name__ == '__main__':
    unittest.main()