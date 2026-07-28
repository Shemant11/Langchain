from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline,HuggingFaceEndpoint
from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct"
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    pipeline_kwargs=dict(temperature=0.5)
)
model=ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'India'})

print(result)

chain.get_graph().print_ascii()