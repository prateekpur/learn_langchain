import os
from dotenv import find_dotenv, load_dotenv
import openai
import chromadb
from chromadb.utils import embedding_functions
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain import LLMChain, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings()

sample_string_1 = "The sky is blue."
sample_string_2 = "Cats are known for their independence and playful behavior."
sample_string_3 = "I believe that learning new languages opens up many opportunities."
sample_string_4 = "The earth is brown"


embed1 = embeddings.embed_query(sample_string_1)
embed2 = embeddings.embed_query(sample_string_2)
embed3 = embeddings.embed_query(sample_string_4)

with open("/Users/prateekpuri/ai_agent/miscllaneous1978/coursera/learn_langchain/langchain/data/dream.txt") as paper:
    speech = paper.read()

text_splitter_recursive = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 20, length_function = len)
texts = text_splitter_recursive.create_documents([speech])

#Chroma.from_documents(documents = texts, embedding = embed1)

client = chromadb.HttpClient(host="localhost", port=8000)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="docs_collection",
    embedding_function=openai_ef
)

collection.add(
    documents=[doc.page_content for doc in texts],
    metadatas=[doc.metadata if doc.metadata else {"source": "manual"} for doc in texts],
    ids=[f"doc_{i}" for i in range(len(texts))]
)

results = collection.query(
    query_texts=["Who wrote I have a Dream?"],
    n_results=2
)

print("Query results:", results)