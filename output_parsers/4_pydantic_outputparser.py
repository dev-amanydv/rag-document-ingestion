from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task="text-generation"
)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description='City of the person')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""Generate a fictional {place} person.

Return ONLY a JSON object.

Do NOT explain anything.
Do NOT repeat the schema.
Do NOT use markdown.\n{format_instruction}""",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
prompt = template.invoke({'place': "Indian"})
print(f"prompt: {prompt}")
model = ChatHuggingFace(llm=llm)

result = model.invoke(prompt)
print(f"=================\n{result.content}")
final_result = parser.parse(result.content)
print(f"============\n{final_result}")