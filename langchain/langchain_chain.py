import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

def generate_story(location, character, language) :
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature = 0.9, model = llm_model, openai_api_key=OPENAI_API_KEY)
    open_ai = OpenAI(temperature = 0.9, openai_api_key=OPENAI_API_KEY)
    prompt_generate_story = PromptTemplate(input_variables=["location", "character"],
                            template = """ Write a short story based on location {location} and main character {character}
                            STORY : """)
    chain_gen_story = LLMChain(llm=open_ai, prompt=prompt_generate_story, output_key="story")
    prompt_translate = PromptTemplate(input_variables=["story", "language"],
                            template="Translate the following story into {language}:\n\n{story}")
                            #template = """ Translate {story} into {language} """)
    chain_translate = LLMChain(llm=open_ai, prompt=prompt_translate, output_key="translated")

    overall_chain = SequentialChain(chains = [chain_gen_story, chain_translate],
                                    input_variables=["location", "character", "language"],
                                    output_variables=["story", "translated"])

    response = overall_chain({"location": location,
                            "character": character,
                            "language": language})
    return response

if __name__ == "__main__":
    response = generate_story("Toronto", "Hulk", "Hindi")
    print(response['translated'])
