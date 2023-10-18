import unittest
import json
from yandex_chain import YandexEmbeddings, YException

class TestYandexEmbeddings(unittest.TestCase):

    def test_embed_document_and_create_from_file(self):
        YGPTE = YandexEmbeddings(config="tests/config.json")
        res = YGPTE.embed_document('Hello, world')
        self.assertEqual(len(res), 256)

    def test_embed_query_and_create_from_apikey(self):
        with open('tests/config.json','r',encoding='utf-8') as f:
            js = json.load(f)
        YGPTE = YandexEmbeddings(folder_id=js['folder_id'],api_key=js['api_key'])
        res = YGPTE.embed_query('Hello, world')
        self.assertEqual(len(res), 256)

    def test_create_embeddings_with_sleep_interval(self):
        with open('tests/config.json','r',encoding='utf-8') as f:
            js = json.load(f)
        YGPTE = YandexEmbeddings(folder_id=js['folder_id'],api_key=js['api_key'],sleep_interval=0.5)
        res = YGPTE.embed_query('Hello, world')
        self.assertEqual(len(res), 256)

    def test_embed_documents(self):
        YGPTE = YandexEmbeddings(config="tests/config.json")
        res = YGPTE.embed_documents(['Hello', 'world'])
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 256)

    def wrong_auth(self):
        YGPTE = YandexEmbeddings(folder_id='xxxxxxx',iam_token='xxxxxxx')
        res = YGPTE.embed_query('Hello, world')

    def test_wrong_auth(self):
        self.assertRaises(YException,self.wrong_auth)

if __name__ == '__main__':
    unittest.main()