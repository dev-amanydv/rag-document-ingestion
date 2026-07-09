from typing import TypedDict, Annotated, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_GPT_API_VERSION")

model = init_chat_model(
    "azure_openai:gpt-5-mini"
)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the feedback')

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negatice \n {feedback} \n {format_instruction}",
    partial_variables={'format_instruction': parser2.get_format_instructions()},
    input_variables=['feedback']
)

classify_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

conditional_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model  | parser1),
    RunnableLambda(lambda x:'could not find')
)
print("========")
chain = classify_chain | conditional_chain
final_result = chain.invoke({'feedback': 'This is a wonderful product'})
print(final_result)