import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MultiPromptChain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain import LLMChain, PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)

# Import necessary libraries

# Step 1: Create a default prompt
default_prompt = PromptTemplate(template="Please provide a general answer to: {input}")

# Step 2: Create the default chain
physics_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate.from_template("You are a physics expert. Answer: {question}")
)

math_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate.from_template("You are a math expert. Solve: {question}")
)

chemistry_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate.from_template("You are a chemistry expert. Explain: {question}")
)

default_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate.from_template("General answer: {question}")
)

def route(inputs):
    if isinstance(inputs, dict):
        q = inputs.get("question")
    else:
        q = inputs  # fallback if it's just a string

    if isinstance(q, dict):
        q = str(q)

    if q is None:
        q = ""  # fallback if missing

    # Use LLM to decide which chain to pick
    classifier_prompt = f"""
    You are an expert classifier. 
    Decide the category for the question below: 'physics', 'math', 'chemistry', or 'general'.

    Question: {q}

    Only return one word: physics, math, chemistry, or general
    """

    llm_output = chat.predict(classifier_prompt).strip().lower()
    if "physics" in llm_output:
        return physics_chain
    elif "math" in llm_output:
        return math_chain
    elif "chemistry" in llm_output:
        return chemistry_chain
    else:
        return default_chain


router = (
    {"question": RunnablePassthrough()}  # just pass the input
    | RunnableLambda(route)               # route to correct chain
)

print(router.invoke({"question": "What is Newton's second law?"}))
print(router.invoke({"question": "Calculate 12 * (3 + 4)"}))