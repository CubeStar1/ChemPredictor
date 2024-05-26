# Standard Libraries
import os
import glob
import tempfile

# Image Processing Libraries
import PIL.Image

# Text-to-Speech Libraries
from gtts import gTTS

# Streamlit Libraries
import streamlit as st

# Google Generative AI Libraries
import google.generativeai as genai


BASE_PROMPT_VISION = "What is this compound and what are some of its properties? Answer it as short as possible and include all equations."
BASE_PROMPT = "Provide concise and informative responses to my questions. Use bullet points to list properties and equations sparingly. Avoid repeating information and ensure that your responses are tailored to my specific questions"

def stream_response():
    st.session_state.stream_response = not st.session_state.stream_response

def text_to_speech(text, lang='en', slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    return tts
def chat_page_fn(model):

    if "stream_response" not in st.session_state:
        st.session_state.stream_response = False
    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_API_KEY = ""
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    if "prediction_df_html" not in st.session_state:
        st.session_state.prediction_df_html = []
    with st.expander("Recent Prediction", expanded=True):
        if st.session_state.prediction_df_html != []:
            st.markdown(f'<div id="" style="overflow:scroll; height:300px; padding-left: 20px; ">{st.session_state.prediction_df_html[-1]}</div>',
                                   unsafe_allow_html=True)
    st.markdown('---')

    # Set a default model
    if "genai_model" not in st.session_state:
        st.session_state["genai_model"] = "gemini-pro"

    # Set a default API key
    if st.session_state["GOOGLE_API_KEY"] == "":
        with st.expander("Set Google API Key"):
            GOOGLE_API_KEY = st.text_input("Enter your Google API Key")
        st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        genai.configure(api_key=GOOGLE_API_KEY)


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # with st.container(border=True, height=500):
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "user":
            if message["parts"] != BASE_PROMPT_VISION:
                with st.chat_message(message["role"]):
                    st.markdown(message["parts"])
        if message["role"] == "model":
            with st.chat_message("assistant"):
                st.markdown(message["parts"])

    if st.session_state.message_count == 0 and st.session_state.GOOGLE_API_KEY != "":
        img_path = glob.glob('./images/*.png')[0]
        print(img_path)
        img = PIL.Image.open(img_path)
        model_vision = genai.GenerativeModel('gemini-pro-vision')
        message_vision = [BASE_PROMPT_VISION, img]
        response_vision = model_vision.generate_content(message_vision, safety_settings={'HATE_SPEECH':'block_none'})
        with st.chat_message("assistant"):
            st.markdown(response_vision.text)
            # tts = text_to_speech(response_vision.text, lang='en', slow=False)
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            #     tts.save(fp.name)
            #     st.audio(fp.name, format="audio/mp3")

        st.session_state.messages.append({"role": "user", "parts": BASE_PROMPT_VISION})
        st.session_state.messages.append({"role": "model", "parts": response_vision.text})
        #st.markdown(st.session_state.messages)
        #st.markdown(message_count)
        st.session_state.message_count += 1


    # Accept user input
    if st.session_state.GOOGLE_API_KEY != "":
        if prompt := st.chat_input("What is up?"):

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "parts": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                messages = [{"role": m["role"], "parts": [m["parts"]]} for m in st.session_state.messages]
                # with st.spinner("Thinking..."):
                #model.generate_content()

                response = model.generate_content(messages, stream=st.session_state.stream_response, safety_settings={'HATE_SPEECH':'block_none','HARASSMENT':'block_none'})

                if st.session_state.stream_response:
                    for chunk in response:
                        st.markdown(chunk.text)
                else:
                    st.write(response.text)
            st.session_state.messages.append({"role": "model", "parts": response.text})
        if st.toggle("Toggle Stream"):
            st.session_state.stream_response = not st.session_state.stream_response
        col1, col2 = st.columns([1, 3])
        with col1:
            tts_button = st.button("Listen to the response", use_container_width=True)
        if tts_button:
            print(st.session_state.messages[-1]['parts'])
            tts = text_to_speech(st.session_state.messages[-1]['parts'], lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                with col2:
                    st.audio(fp.name, format="audio/mp3")

