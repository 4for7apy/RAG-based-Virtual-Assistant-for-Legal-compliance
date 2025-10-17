import os
import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import time
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

# Progress bar and success message during loading
# with st.empty():
#     for p_comp in range(25):
#         time.sleep(.1)
#         st.progress(p_comp + 1, text='Loading..........')
#     st.success('Always Ready to Serve you')

# Initialize OpenAI API and Qdrant client
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

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

# Initialize retriever and QA system
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Display logo and title
LOGO_IMAGE = "Sih logo.png"
st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 ;
        font-size:50px ;
        color: #f0000 ;
    }
    .logo-img {
        float:right;
        border-radius:25px;
        margin:15px ;
        width: 130px; 
        height: 130px;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">Always Ready, to help you out</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Initialize recognizer and microphone for voice input
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Function to recognize speech from microphone
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        st.write("Say something...")
        audio = recognizer.listen(source)
    try:
        response = recognizer.recognize_google(audio)
        st.write(f"You said: {response}")
        return response
    except sr.RequestError:
        st.write("API unavailable")
    except sr.UnknownValueError:
        st.write("Unable to recognize speech")
    return ""

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    tts_audio = BytesIO()
    tts.save(tts_audio, format="mp3")
    tts_audio.seek(0)
    return tts_audio

# Sidebar option for voice input
st.sidebar.write("## Voice-Enabled Chatbot 🎤")
use_voice = st.sidebar.checkbox("Use Voice Input 🎙️")

# Handle user input (either voice or text)
if use_voice:
    user_input = recognize_speech_from_mic(recognizer, microphone)
else:
    user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    response = qa.run(user_input)

    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    if use_voice:
        tts_audio = text_to_speech(msg)
        st.audio(tts_audio)
