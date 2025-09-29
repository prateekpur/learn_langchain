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
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
loader = PyPDFLoader("/Users/prateekpuri/ai_agent/miscllaneous1978/coursera/learn_langchain/langchain/data/okta-last-quarter.pdf")
pages = loader.load()
print(len(pages))
print(pages[0])
