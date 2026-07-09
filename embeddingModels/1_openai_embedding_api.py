from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_EMBEDDING_API_VERSION")

embedding = AzureOpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result = embedding.embed_query("What is the capital of India")
print(result)

