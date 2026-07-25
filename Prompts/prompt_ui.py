from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

st.header("Chat with LLMs")

user_input = st.text_input("Ask what you want to know")

if st.button("Search"):
    result = model.invoke(user_input)
    st.write(result.content)