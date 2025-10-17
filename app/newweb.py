import streamlit as st
import openai
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain import hub
import os
from dotenv import load_dotenv 
load_dotenv() 


openai.api_key=(os.getenv("MY_KEY"))


# Load the latest version of the prompt
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

# Load documents
raw_documents = TextLoader(r"D:\Infosys Chatbot Development\scrapped.txt", encoding='utf-8').load()
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Store splits in vector store
vectorstore = Qdrant.from_documents(documents=documents, embedding=OpenAIEmbeddings(openai_api_key=openai.api_key))

# Language Model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key)

# RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt}
)


def handle_chat():
    user_message = st.session_state.user_input.strip()
    if user_message:
        # Get response from the chat model
        response_data = qa_chain.invoke({"query": user_message})
        # Check if result is available and set the response
        if response_data.get("result") and response_data["result"].strip():
            response = response_data["result"]
        else:
            response = "I don't know."
        
        # Update chat history
        st.session_state.chat_history.append(("You", user_message))
        st.session_state.chat_history.append(("Bot", response))

        # Clear the input box after processing
        st.session_state.user_input = ""



# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title('Welcome to the Infosys Chatbot')
st.write("Ask your questions here!")

# Display chat history
for role, message in st.session_state.chat_history:
    st.text_area(f"{role}:", value=message, height=100, max_chars=None)

# UI for sending new messages
user_input = st.text_input("Type your question about Infosys here:", key="user_input")
send_button = st.button("Send", on_click=handle_chat)
