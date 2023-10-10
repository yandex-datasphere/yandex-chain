# yandex-chain - LangChain-compatible integrations with YandexGPT and YandexGPT Embeddings

This library is community-maintained Python package that provides support for [Yandex GPT](https://cloud.yandex.ru/docs/yandexgpt/) LLM and Embeddings for [LangChain Framework](https://www.langchain.com/).

> Currently, Yandex GPT is in preview stage, so this library may occasionally break. Please use it at your own risk!

## What's Included

The library includes the following two main classes:

* **YandexLLM** is a class representing [YandexGPT Text Generation](https://cloud.yandex.ru/docs/yandexgpt/api-ref/TextGeneration/).
* **YandexEmbeddings** represents [YandexGPT Embeddings](https://cloud.yandex.ru/docs/yandexgpt/api-ref/Embeddings/) service.

## Usage

You can use `YandexLLM` in the following manner:

```python
from yandex_chain import YandexLLM

LLM = YandexLLM(folder_id="...", api_key="...")
print(LLM("How are you today?"))
```

You can use `YandexEmbeddings` to compute embedding vectors:

```python
from yandex_chain import YandexEmbeddings

embeddings = YandexEmbeddings(...)
print(embeddings("How are you today?"))
```

## Authentication

In order to use Yandex GPT, you need to provide one of the following authentication methods, which you can specify as parameters to `YandexLLM` and `YandexEmbeddings` classes:

* A pair of `folder_id` and `api_key`
* A pair of `folder_id` and `iam_token`
* A path to [`config.json`](tests/config_sample.json) file, which may in turn contain parameters listed above in a convenient JSON format.

## Complete Example

A pair of LLM and Embeddings are a good combination to create problem-oriented chatbots using Retrieval-Augmented Generation (RAG). Here is a short example of this approach, inspired by [this LangChain tutorial](https://python.langchain.com/docs/expression_language/cookbook/retrieval).

To begin with, we have a set of documents `docs` (for simplicity, let's assume it is just a list of strings), which we store in vector storage. We can use `YandexEmbeddings` to compute embedding vectors:

```python
from yandex_chain import YandexLLM, YandexEmbeddings
from langchain.vectorstores import FAISS

embeddings = YandexEmbeddings(config="config.json")
vectorstore = FAISS.from_texts(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

We can now retrieve a set of documents relevant to a query:

```python
query = "Which library can be used to work with Yandex GPT?"
res = retriever.get_relevant_documents(query)
```

Now, to provide a full-text answer to the query, we can use LLM. We will prompt the LLM, giving it retrieved documents as a context, and the input query, and ask it to answer the question. This can be done using LangChain *chains*:

```python
from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = YandexLLM(config="config.json")

chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)
```

This chain can now answer our questions:
```python
chain.invoke(query)
```

## Testing

This repository contains some basic unit tests. To run them, you need to place a configuration file `config.json` with your credentials into `tests` folder. Use `config_sample.json` as a reference. After that, please run the following at the repository root directory:

```bash
python -m unittest discover -s tests
```

## Credits

* This library has originally been developed by [Dmitri Soshnikov](https://soshnikov.com).