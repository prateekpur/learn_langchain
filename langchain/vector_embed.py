import os
from dotenv import find_dotenv, load_dotenv
import openai
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
#from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain import LLMChain, PromptTemplate
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

client = chromadb.HttpClient(host="localhost", port=8000)

collection = client.get_or_create_collection("my_collection")
collection.add(
    documents=["Chroma via docker-compose!"],
    ids=["doc1"]
)

print(collection.query(query_texts=["docker"], n_results=1))

#similarity = np.dot(embed1, embed2)
#print(similarity * 100)

#similarity = np.dot(embed1, embed3)
#print(similarity * 100)