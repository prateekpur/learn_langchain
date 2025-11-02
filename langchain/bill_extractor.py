import os
import json
import re
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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature = 0.0)


def extract_data(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            # Replace multiple newlines with one space
            page_text = re.sub(r'\s*\n\s*', ' ', page_text)
            text += page_text.strip() + " "
    return text.strip()


def parse_data(file):
    file_text = extract_data(file)
    template = """ Extract these values : invoice id , description, issue date, unit price, amount, bill for, from, terms from: {pages}.
    Expected output is a json {{"invoice_id": "12345", "description": "Monthly utility bill", "issue_date": "2023-01-15","due_date": "2023-02-15",
    "amount": "$150.00", "from": "Utility Company","to": "John Doe"}}"""
    prompt = PromptTemplate(input_variables=["pages"], template=template)
    llm1 = "gpt-4o-mini"
    llm = OpenAI(temperature=0.7)
    res = llm(prompt.format(pages=file_text))
    return res
#invoice id , description, issue date, unit price, amount, bill for, from, terms

def create_pandas(json_str):
    # Create a DataFrame from the JSON data
    data = json.loads(json_str)
    df = pd.DataFrame([data])
    print(df)
    return df

if __name__ == "__main__":
    response = parse_data("/Users/prateekpuri/Documents/utility/alectra-june.pdf")
    print(response)
    create_pandas(response)
    #print(response)
    # Create a DataFrame from the JSON data
    #create_pandas(response)

# Display the DataFrame
