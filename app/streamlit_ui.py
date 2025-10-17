import os
import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import time

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# with st.empty():
#     for p_comp in range(25):
#         time.sleep(.1)
#         st.progress(p_comp + 1, text= 'Loading..........')
#     st.success('Aagya Kijiye Hujur')

# openai_api_key = os.getenv("OPENAI_API_KEY")
# if openai_api_key is None:
#     raise ValueError("The OPENAI_API_KEY environment variable is not set.")
st.sidebar.write("## CHAT HISTORY")


llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo', api_key=os.environ['OPENAI_API_KEY'])


client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_LOCALHOST"),
        api_key=os.getenv("OPENAI_API_KEY"),
        prefer_grpc=False
    )
collection_name = os.getenv("QDRANT_LOCAL_COLLECTION")


vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_LOCAL_COLLECTION"),
        embeddings=OpenAIEmbeddings()
)
#Initialize retriever and QA system
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)




import base64

LOGO_IMAGE = "Logo1.jpg"

st.markdown(
    """
    <style>
    .container {
        display: flex;
        
    }
    .logo-text {
        font-weight:700 ;
        font-size:40px ;
        color: #f0000 ;
        
        
    }
    .logo-img {
        float:right;
        border-radius:25px;
        margin:15px ;
        width: 60px; 
        height: 60px;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">How Can I Help You Dude!</p>
    </div>
    """,
    unsafe_allow_html=True
)



if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]



for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    response = qa.run(question)
    

    msg = response  
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

