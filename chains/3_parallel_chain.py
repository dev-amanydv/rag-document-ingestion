from typing import TypedDict, Annotated, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_GPT_API_VERSION")

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

model = init_chat_model(
    "azure_openai:gpt-5-mini"
)

prompt1 = PromptTemplate(
    template="Generate short and simple note from following text: \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate the 5 short ques/ans from the follwing text: \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document. \n Notes -> {notes} and \n {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model | parser,
        'quiz': prompt2 | model | parser
    }
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
The Particle Model of Matter and Density1. The States of MatterMatter exists in three primary states: solid, liquid, and gas. Each state is defined by the arrangement and movement of its particles. In a solid, particles are tightly packed in a regular lattice and vibrate in place, giving the solid a fixed shape and volume. In a liquid, particles are close together but can move past one another, allowing the liquid to flow and take the shape of its container. In a gas, particles are widely spaced and move randomly at high speeds, meaning gases have no fixed shape or volume.2. Density of MaterialsDensity is a fundamental physical property defined as the mass per unit volume of a substance. The mathematical equation for density is \(\rho = \frac{m}{V}\), where ρ represents density, m is mass (measured in kilograms or grams), and V is volume (measured in cubic meters or cubic centimeters). Materials with tightly packed particles, such as most metals, have a high density. Conversely, materials with large spaces between particles, such as gases or certain types of wood, have a low density.3. Changes of StateTransitions between states of matter are physical changes that do not alter the chemical composition of the substance. These changes depend on the addition or removal of thermal energy. For instance, when heat is added to a solid, its particles vibrate faster until they break free from their rigid structure—a process known as melting. When thermal energy is removed from a gas, particles slow down and draw closer together, undergoing condensation to become a liquid.
"""

result = chain.invoke({
    'text': text
})

print(result)