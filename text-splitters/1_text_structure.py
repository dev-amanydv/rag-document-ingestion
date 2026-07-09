from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_docling.loader import DoclingLoader

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_GPT_API_VERSION")

model = init_chat_model(
    "azure_openai:gpt-5-mini"
)

loader = DoclingLoader(file_path="text-splitters/Text_Splitter_Test_Document.docx")
docs = loader.load()
print(len(docs))
# print("-----------")
# print(docs[0].page_content)
# print(docs[0].metadata)
# print("-----------")
# print(docs[1].page_content)
# print(docs[1].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separators=['/n']
)

splitted_docs = text_splitter.split_documents(docs)
print(f"length of splitted docs: {len(splitted_docs)}")
print(f"doc1: {splitted_docs[0].page_content}\n-------\n{splitted_docs[1].page_content}")







