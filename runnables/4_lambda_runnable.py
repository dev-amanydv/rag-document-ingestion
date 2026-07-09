
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
    template="Generate a joke on the topic of {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='You are an expert llm trained heavily on X algorithm and data. Generate an X post for explanation of following joke: \n {joke}',
    input_variables=['joke']
)

prompt3 = PromptTemplate(
    template='You are an expert llm trained heavily on Linkedin algorithm and data. Generate an Linkedin post for explanation of following joke: \n {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

def print_output(output: str):
    print(output)
    print("===========")
    return output

joke_generator_chain = RunnableSequence(prompt1, model, parser, print_output)

parallel_chain = RunnableParallel({
    'linkedin': RunnableSequence(prompt1, model, parser, print_output),
    'lambda': RunnableLambda(lambda x: print(f'inside runnable lambda: {x}'))
})

result1 = parallel_chain.invoke({'topic': 'runnable langchain'})
print(result1)
print('=============')
final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

final_result = final_chain.invoke({'topic': 'joke'})
print(final_result)