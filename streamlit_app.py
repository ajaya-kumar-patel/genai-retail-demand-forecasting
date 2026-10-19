import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.genai.graph import graph


st.set_page_config(
    page_title="Retail GenAI Assistant",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Retail GenAI Assistant")
st.caption("Ask questions about sales, products, and business performance.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question
if prompt := st.chat_input("Ask a sales analytics question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                # Convert Streamlit chat history into LangChain messages
                conversation = []
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        conversation.append(
                            HumanMessage(content=message["content"])
                        )
                    elif message["role"] == "assistant":
                        conversation.append(
                            AIMessage(content=message["content"])
                        )

                # Pass the full conversation to LangGraph
                result = graph.invoke({
                    "messages": conversation
                })

                answer = result["messages"][-1].content
                st.markdown(answer)

            except Exception:
                answer = (
                    "Sorry, I couldn't process that question. "
                    "Please check that Ollama and the required "
                    "database services are running."
                )
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.title("About")
st.sidebar.write("Powered by LangGraph, Ollama, and MySQL.")
