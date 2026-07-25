from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model= ChatGoogleGenerativeAI(model='gemini-3.6-flash')
result=model.invoke("What is the capital of India?")
q1=model.invoke("Hii! who are you?")
q2=model.invoke("who is your owner?")
q3=model.invoke("what i can do with you?")
print(result.content[0]["text"])
print(q1.content[0]["text"])
print(q2.content[0]["text"])
print(q3.content[0]["text"])    
