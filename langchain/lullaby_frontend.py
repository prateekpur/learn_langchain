import os
import langchain_chain
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
import streamlit as st


st.set_page_config(page_title="Generate lollaby", layout="centered")

def main():
    st.title("Test Lollaby")
    st.header("start")
    location = st.text_input(label="Location")
    character = st.text_input(label="Character")
    lang = st.text_input(label="Language")
    submit_button = st.button("Submit")

    if location and character and lang:
        if submit_button:
            response = langchain_chain.generate_story(location=location, character=character, language=lang)
            with st.expander("English version"):
                st.write(response['story'])
            with st.expander(f"{lang} version"):
                st.write(response['translated'])
            


if __name__ == "__main__":
    main()