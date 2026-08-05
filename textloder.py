from langchain_community.document_loaders import TextLoader
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write the summary for the startups \n {startups}',
    input_variables=['startups']
)
parser=StrOutputParser()

loader=TextLoader("startups.txt")
docs = loader.load()
print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser
print(chain.invoke({'startups':docs[0].page_content}))