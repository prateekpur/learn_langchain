import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.chains import MultiPromptChain
from langchain.document_loaders import PyPDFLoader
from langchain import LLMChain, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature = 0.0)

prompt = PromptTemplate(input_variables=["query"], template = "{query}")
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_tool = Tool(name = "Language Model", func=llm_chain.run, description="General Tool")

llm_math = LLMMathChain.from_llm(llm = llm)
math_tool = Tool(name="Calculator", func=llm_math.run, description="Useful for math questions")
tool = load_tools(['llm-math'], llm=llm)
tool.append(llm_tool)


agent = initialize_agent(agent="zero-shot-react-description", tools=tool, llm=llm, verbose=True, max_iterations=3)
query = "If i have 12 eggs and a friend has 5. How many eggs do we have in total"
print("Agent  !@#!@#!@#!@#", agent.agent.llm_chain.prompt.template)

result = agent(query)
print(result)

