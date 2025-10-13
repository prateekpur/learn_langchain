import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain import LLMChain, PromptTemplate, WikiPedia
from langchain.memory import ConversationBufferMemory
from langchain.agents.react.base import DocstoreExplorer

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature = 0.0)

prompt = PromptTemplate(input_variables=["query"], template = "{query}")
docstore = DocstoreExplorer(WikiPedia())
llm_chain = LLMChain(llm=llm, prompt=prompt)
llm_tool = Tool(name = "Language Model", func=llm_chain.run, description="General Tool")

llm_math = LLMMathChain.from_llm(llm = llm)
math_tool = Tool(name="Calculator", func=llm_math.run, description="Useful for math questions")
tool = load_tools(['llm-math'], llm=llm)
tool.append(llm_tool)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(agent="conversational-react-description", tools=tool, llm=llm, verbose=True, max_iterations=3, memory=memory)
query = "A persom was born in 2011. What would be his age in 2023 ?"
query_2 = "What ould be his age 100 years from now?"
print("Agent  !@#!@#!@#!@#", agent.agent.llm_chain.prompt.template)

result = agent(query)
result_2 = agent(query_2)
print(result)
print(result_2)

