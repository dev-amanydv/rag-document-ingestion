import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class FakeLLM(Runnable):
    def __init__(self):
        pass
        
    def invoke(self, str):
        responses = [
            'You are a fool. get off my dick mf!',
            'AI stands for artificial Intelligence',
            'Mujhe please chod do!!'
        ]
        return {'response': random.choice(responses)}
    def predict(self, str):
        responses = [
            'You are a fool. get off my dick mf!',
            'AI stands for artificial Intelligence',
            'Mujhe please chod do!!'
        ]
        return {'response': random.choice(responses)}


llm = FakeLLM()

print(llm.invoke('ddnbrb'))

class FakeLLMTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
    
    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    
    def format(self, input_dict):
        return self.template.format(**input_dict)

template = FakeLLMTemplate(
    template="Write a poem about {topic}",
    input_variables=['topic']
)

class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

chain = RunnableConnector([template, llm])
result = chain.invoke({ 'topic': 'dndbd'})
print(result)

class LLMChain:
    def __init__(self, llm, template):
        self.llm = llm
        self.template = template
    
    def run(self, input_dict):
        final_prompt = self.template.format(input_dict)
        result = self.llm.invoke(final_prompt)
        return result['response']
    
chain = LLMChain(llm=llm, template=template)
final = chain.run({ 'topic': 'uneb'})
print(final)
