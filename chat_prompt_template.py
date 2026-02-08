from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expart'),
    ('human','Explain the simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'six'})

print(prompt)