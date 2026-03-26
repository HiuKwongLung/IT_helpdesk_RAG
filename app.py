import streamlit as st
from rag import get_answer

st.title("IT Helpdesk Chatbot")

query = st.text_input("Ask anything:")
if query:
    answer = get_answer(query)
    st.write(answer)