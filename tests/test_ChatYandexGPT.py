import unittest
from yandex_chain import ChatYandexGPT, YException
from langchain.schema import HumanMessage,AIMessage

class TestChatYandexGPT(unittest.TestCase):

    def test_create_from_file(self):
        YGPT = ChatYandexGPT(config="tests/config.json")
        res = YGPT([HumanMessage(content='Imagine no possessions...')])
        self.assertTrue(isinstance(res,AIMessage))
        self.assertGreater(len(res.content), 10)
        self.assertGreater(YGPT.totalTokens,0)
        self.assertGreater(YGPT.completionTokens,0)
        self.assertGreater(YGPT.inputTextTokens,0)

    def wrong_auth(self):
        YGPT = ChatYandexGPT(folder_id='xxxxxxx',iam_token='xxxxxxx')
        res = YGPT([HumanMessage(content='Imagine no possessions...')])

    def test_wrong_auth(self):
        self.assertRaises(YException,self.wrong_auth)

if __name__ == '__main__':
    unittest.main()