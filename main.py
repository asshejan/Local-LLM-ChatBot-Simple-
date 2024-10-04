from langchain_ollama import OllamaLLM #Python package that integrates the Ollama platform
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversational history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to ChatBot! type exit to quit")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "question": user_input})
        context += f"\nUser: {user_input}\nAI: {result}"
        print("Bot: ", result)

if __name__ == "__main__":
    handle_conversation()




