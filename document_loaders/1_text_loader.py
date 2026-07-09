from langchain_docling.loader import DoclingLoader

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
    template="Write a summary for the following text:\n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

loader = DoclingLoader(file_path='document_loaders/data/cricket.txt')

docs = loader.load()

chain = prompt1 | model | parser
result = chain.invoke({'text': docs[0].page_content})
print(result)