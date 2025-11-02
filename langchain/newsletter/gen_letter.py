import os
import json
import requests
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

embeddings = OpenAIEmbeddings()
llm_model = "gpt-4o-mini"
chat = ChatOpenAI(temperature = 0.0, model = llm_model, openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature = 0.7)


def search_serper(query) :
    search = GoogleSerperAPIWrapper(k=5, type="search")
    # Perform a search
    data = search.results(query)
    #print(f"Results === >{data}")
    #print(data.keys())
    return data

def pick_best_url(resp_json, query) :
    response_str = json.dumps(resp_json)
    #print(f"Response String ===> {response_str}")
    template = """
        You are good at finding relevant urls's and topics

        Query response : {response_str}
        Above is the list of searh articles for query {query}

        Please choose the best 3 artcles from the list and return only an array of the link. Do not incllude anything else
        Also make sure the articles are recent and not too old.
        If the file or url is invalid show www.google.com
        Return them as a JSON list of valid URLs only, no commentary."""
    prompt_template = PromptTemplate(input_variables=["response_str", "query"], template=template)
    article_chooser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    urls = article_chooser_chain.run(response_str=response_str, query=query)

    url_list = json.loads(urls)
    return url_list

#extract url content and create embeddings
def extract_content(urls) :
    valid_urls = []
    headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
    for url in urls:
        try:
            resp = requests.get(url, allow_redirects=True, headers=headers, timeout=10)
            if resp.status_code == 200:
                valid_urls.append(url)
            else:
                print(f"URL not reachable ({resp.status_code}): {url}")
        except Exception as e:
            print(f"Error checking URL {url}: {e}")

    # Load content
    loader = UnstructuredURLLoader(
        urls=valid_urls,
        continue_on_failure=True,
        headers={"User-Agent": "Mozilla/5.0"},
        requests_kwargs={"timeout": 60}
    )
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(data)
    # Extract page content
    texts = [doc.page_content for doc in split_docs if doc.page_content.strip()]
    print(f"Text before creating embeddings {texts}")
    if not texts:
        print("No text content extracted from URLs.")
        return []

    # Split text into chunks
    db = FAISS.from_documents(split_docs, embeddings)

    return db

def summarizer(db, query, k=4):
    docs = db.similarity_search(query, k=4)
    docs_page_content = " ".join(d.page_content for d in docs)
    template = """
        {docs}
        As a good journalist summarize the text to create newsletter around {query}
        News letter should have a format like Tim Ferris "5 point letter".
        
        Please follow these guidelines:
        1. Make sure content is engaging , informative with good data
        2. Make sure the content is not too long
        3. The content should address {query} topic very well
        4. The content needs to be good and informative
        5. The content needs to be written in a way that is easy to read and digest
        6. The content neeeds to give the audience actionable insight
"""
    prompt_template = PromptTemplate(input_variables=["docs", "query"], template=template)
    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = summarizer_chain.run(docs=docs_page_content, query=query)
    return response.replace("\n", "")

def generate_letter(summaries, query) :
    summaries_str = str(summaries)
    template = """
        {summaries_str}
        As a good write use the text above as context for {query} to write a newletter to be sent to subscribers about {query}
        News letter should have a format like Tim Ferris "5 point letter".
"""
    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    letter_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = letter_chain.run(summaries_str=summaries_str, query=query)
    print(response)
    return response


def main() :
    query = "US open tennis 2025 results"
    resp = search_serper(query)
    #print(resp)
    response_str = json.dumps(resp)
    #print(f"Response_str {response_str}")

    url_list = pick_best_url(resp, query)
    #url_list1 = list(url_list.values())
    print(url_list)
    data = extract_content(url_list)
    summary = summarizer(data, query)
    generate_letter(summaries=summary, query=query)

    #print(resp)

if __name__ == "__main__":
    main()
