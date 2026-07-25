from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


pipe = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-135M-Instruct"
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    pipeline_kwargs=dict(temperature=0.5,max_new_tokens=20)
)

model=ChatHuggingFace(llm=llm)
result=model.invoke("What is the capital of India?")
q1=model.invoke("Hii! who are you?")
print(result.content)
print(q1.content)

