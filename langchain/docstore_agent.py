import os
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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature = 0.0)

prompt = PromptTemplate(input_variables=["query"], template = "{query}")
docstore = DocstoreExplorer(Wikipedia())
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_tool = Tool(name = "Language Model", func=llm_chain.run, description="General Tool")

tool = load_tools(['llm-math'], llm=llm)
tool.append(llm_tool)

tools = [Tool(name="Search", func=docstore.search, description="Search wikipedia"), 
         Tool(name="Lookup", func=docstore.lookup, description="Lookup wikipedia")]

docstoreAgent = initialize_agent(tools,llm,agent="react-docstore",verbose=True,max_iterations=5)

query = "What was Bach's last piece he wrote"
print(docstoreAgent.run(query))