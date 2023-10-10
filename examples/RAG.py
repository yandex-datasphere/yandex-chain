from yandex_chain import YandexLLM, YandexEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

docs = [
    "I am a chat-bot that tries to preted to be intelligent",
    "I am only a few days old",
    "My name is RAG, which stands for Retrieval-Augmented Generation",
    "I am not fully intelligent, although I can very well preted to be"
]

print(" + Indexing documents")
embeddings = YandexEmbeddings(config="tests/config.json")
vectorstore = FAISS.from_texts(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

query = "How old are you?"

print(f"+ Query: {query}")

print(" + Getting relevant documents")
res = retriever.get_relevant_documents(query)

for x in res:
    print(f"{x}\n------\n")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = YandexLLM(config="tests/config.json")

chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

print(" + Running LLM Chain")

res = chain.invoke(query)

print(f" + Answer is: {res}")
