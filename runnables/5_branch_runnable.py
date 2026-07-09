
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_GPT_API_VERSION")

model = init_chat_model(
    "azure_openai:gpt-5-mini"
)

prompt1 = PromptTemplate(
    template="Generate a joke with more than 200 characters on the topic of {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarise the following joke: \n {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

def print_output(output: str):
    print(output)
    print("===========")
    return output

joke_generator_chain = RunnableSequence(prompt1, model, parser, print_output)

def count_length(joke: str):
    length = len(joke)
    print(f"length: {length}")
    return length

conditional_chain = RunnableBranch(
    (lambda x: count_length(x) > 150, RunnableSequence(prompt2, model, parser)),
    (lambda x: count_length(x) < 150, RunnableLambda(lambda x: print("Your joke is already less than 150 chars"))),
    RunnableLambda(lambda x: print('Default conditon ran'))
)

final_chain = RunnableSequence(joke_generator_chain, conditional_chain)

final_result = final_chain.invoke({'topic': 'langchain runnable'})
print(final_result)