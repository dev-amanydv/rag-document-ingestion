from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import str
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task="text-generation"
)

schema = [
    StructuredOutputParser(data)
]

model = ChatHuggingFace(llm=llm)

parser = 

template1 = PromptTemplate(
    template="Give me the name, age and city og a fiction person.  Be strict about the following condition, don't return nothing extra: \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template1.format()
print(prompt)
print("======================")
# result1 = model.invoke(prompt)
# print(result1.content)
# parsed_result = parser.parse(result1.content)
chain = template1 | model | parser 
result = chain.invoke({})

print(result)
