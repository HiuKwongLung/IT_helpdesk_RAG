import streamlit as st
from rag import get_answer

st.title("IT Helpdesk Chatbot")

# store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
query = st.chat_input("Ask anything:")
if query:

    st.session_state.messages.append({"role":"user", "content":query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(query)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})