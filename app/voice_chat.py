import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
import streamlit as st


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


def text_to_speech(text):
    tts = gTTS(text)
    tts_audio = BytesIO()
    tts.save(tts_audio, format="mp3")
    tts_audio.seek(0)
    return tts_audio


# Initialize recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

st.sidebar.write("## Voice-Enabled Chatbot")

use_voice = st.sidebar.checkbox("Use Voice Input")
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


