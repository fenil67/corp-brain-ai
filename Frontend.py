import streamlit as st
import requests

# 1. The Title
st.set_page_config(page_title="CorpBrain Agent", page_icon="🧠")
st.title("🧠 CorpBrain: Talk to your Data")

# 2. Session State (This keeps the chat history alive)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Input Box
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. Call the Backend API (The Engineer's Move)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat", 
                params={"question": prompt}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                # Show the Answer
                message_placeholder.markdown(answer)
                
                # Show the Sources (The "Senior" Touch)
                with st.expander("View Source Documents"):
                    for i, source in enumerate(sources):
                        st.info(f"Source {i+1}: {source}")
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            message_placeholder.error(f"Connection Error. Is the backend running? {e}")