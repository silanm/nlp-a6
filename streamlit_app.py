import streamlit as st
from chatbot import ask

st.title("Ask me about Sila!")

def chat(question):
    answer = ask(question)
    return answer['answer']

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask what!?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(prompt)
            st.markdown(response)
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})