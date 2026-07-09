from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m" 
CYAN = "\033[96m"  
YELLOW = "\033[93m"
GRAY = "\033[90m" 

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_EMBEDDING_API_VERSION")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_video_id(link: str):
    """Get Youtube Video ID"""
    if "https" in link:
        if "youtu.be" in link:
            id = link.split("/")[3].split("?")[0]
            return id
        else:
            url = urlparse(link)
            query_params = parse_qs(url.query)
            id = query_params["v"][0]
            return id


def get_transcript(id: str):
    ytt_api = YouTubeTranscriptApi()
    response = ytt_api.fetch(id, languages=["en"])
    transcript = ""
    for snippet in response:
        transcript += " " + snippet.text
    return transcript


def main():
    print(f"\n{BOLD}{GREEN}YouTube Chatbot{RESET}")
    print(f"{GRAY}Ask me anything about a YouTube video.{RESET}\n")

    link = input(f"{CYAN}Paste the video link{RESET}\n{CYAN}> {RESET}")
    print(f"\n{YELLOW} Reading the video...{RESET}")
    id = get_video_id(link)
    transcript = get_transcript(id)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    docs = text_splitter.create_documents([transcript])

    embedding = AzureOpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)

    vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    template = PromptTemplate(
        template="You're a helpful youtube transcript assistant who solvesuser queries related to a youtube video. You have to generate the answer of the following query strictly based on the context:\n {query}\n\n The relevent contextfrom transcript is:\n{context}",
        input_variables=["query", "context"],
    )
    print(f"\n{GREEN} Ready!{RESET}")
    print(f"{GRAY}Type your question, or type 'exit' to quit.{RESET}")
    print(f"\n{BOLD}{GREEN}Bot:{RESET} Hey, how can I help you?")

    llm = init_chat_model("azure_openai:gpt-5-mini")

    while True:
        query = input(f"\n{BOLD}{CYAN}You:{RESET} ")
        if query.strip().lower() in ("exit", "quit"):
            print(f"\n{BOLD}{GREEN}Bot:{RESET} Bye! \n")
            break

        context = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in context)
        prompt = template.invoke({"query": query, "context": context})

        print(f"{GRAY}...thinking...{RESET}")
        response = llm.invoke(prompt)
        print(f"\n{BOLD}{GREEN}Bot:{RESET} {response.content}")

main()
