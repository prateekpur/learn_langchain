import os
import json
import requests
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain import LLMChain, PromptTemplate, Wikipedia
from langchain.memory import ConversationBufferMemory
from langchain.agents.react.base import DocstoreExplorer
from langchain.document_loaders import PyPDFLoader
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

embeddings = OpenAIEmbeddings()
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature = 0.7)
pf_loader = PyPDFLoader('./docs/PrateekPuri_Resume.pdf')
documents = pf_loader.load()

chain = load_qa_chain(llm, verbose=True)
query = 'He has worked in how many countries ?'
response = chain.run(input_documents=documents, question=query)
print(response)