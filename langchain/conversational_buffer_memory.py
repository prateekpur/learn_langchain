import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAIimport 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

#load_dotenv(find_dotenv())
OPENAI_API_KEY = "<your key>"
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
print(llm.predict("I am XYZ. Who are you ?"))
memory = ConversationBufferMemory()
conversation = ConversationChain(llm = llm, memory = memory, verbose = True)
conversation.predict(input="Hello there" \
", i am xyz")
conversation.predict(input="Why is the sky blue")
conversation.predict(input="If Rayleigh did not exist , will sky be blue")
conversation.predict(input="What is my name ?")

print(memory.load_memory_variables({}))