import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MultiPromptChain
#from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain import LLMChain, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)


with open("/Users/prateekpuri/ai_agent/miscllaneous1978/coursera/learn_langchain/langchain/data/dream.txt") as paper:
    speech = paper.read()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap = 20, length_function = len)
texts = text_splitter.create_documents([speech])
print(texts[0])
print("+++++++++++++++++")

text_splitter_recursive = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 20, length_function = len)
texts = text_splitter_recursive.create_documents([speech])
print(len(texts))

