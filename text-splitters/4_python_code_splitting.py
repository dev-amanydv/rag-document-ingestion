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

docs = """
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence
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
    template='Generate a joke on the topic of {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate the explanation of following joke: \n {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic': 'ai'})
print(result)
"""

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=500,
    chunk_overlap=0,
    language=Language.PYTHON
)

splitted_docs = text_splitter.split_text(docs)
print(f"length of splitted docs: {len(splitted_docs)}\n")
for i, doc in enumerate(splitted_docs):
    print(f"------doc{i+1}-------\n{doc}")







