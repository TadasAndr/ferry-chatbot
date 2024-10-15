import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from backend.llm import LLM, StreamHandler
from backend.vectorstore import load_vector_store
from backend.config import config
import uuid

st.set_page_config(
    page_title="Smiltynės perkelos pokalbiai su DI",
    page_icon="🚢"
)
st.title("Smiltynės perkelos pokalbiai su DI")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'chatbot' not in st.session_state:
    index_name = 'ferry-chatbot'
    vector_store = load_vector_store(index_name)
    st.session_state['chatbot'] = LLM(vector_store)

# Display chat history
for i in range(len(st.session_state['past'])):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(st.session_state['past'][i])
    with st.chat_message("assistant", avatar="🚢"):
        st.markdown(st.session_state['generated'][i])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.past.append(prompt)
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🚢"):
        message_placeholder = st.empty()
        full_response = {"content": ""}

        class CustomStreamHandler(StreamHandler):
            def __init__(self, container):
                super().__init__(container)
                self.container = container

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                full_response["content"] += token
                self.container.markdown(full_response["content"])

        stream_handler = CustomStreamHandler(message_placeholder)

        output = st.session_state['chatbot'].ask(
            prompt,
            st.session_state['session_id'],
            stream_handler
        )

        st.session_state.generated.append(output)