from sign_language_translator import SignLanguageTranslator
import streamlit as st

translator = SignLanguageTranslator()

def translate_to_asl(text):
    return translator.translate(text)

def display_asl_translation(asl_translation):
    for sign in asl_translation:
        st.image(sign)  # Display each image (or video)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you? 🤖"}]
    st.session_state["data"] = []
    st.session_state["chat_count"] = 0  

user_input = st.chat_input("Type your message here... ✍️")

if user_input:
    st.session_state["chat_count"] += 1
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    response = qa.run(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # Translate user input to ASL
    asl_translation = translate_to_asl(user_input)
    display_asl_translation(asl_translation)

    # Save data after each interaction
    save_conversation()

    if st.button("Do you want to ask another question?"):
        st.experimental_rerun()