import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Sidebar - Description and Hugging Face token input
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.write('This chatbot is created using the fine-tuned Llama 2 LLM model from Meta.')

    st.subheader('Authentication')
    if 'HUGGINGFACE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        hf_api_token = st.secrets['HUGGINGFACE_API_TOKEN']
    else:
        hf_api_token = st.text_input('Enter Hugging Face API token:', type='password')
        if not hf_api_token:
            st.warning('Please enter your Hugging Face API token!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['HUGGINGFACE_API_TOKEN'] = hf_api_token

    st.subheader('Models and parameters')
    model_name = "tito92/finance_finetune_model"
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Load the model and tokenizer
@st.cache_resource()
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_token)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating response using the fine-tuned model
def generate_response(prompt_input):
    string_dialogue = "You are a helpful finance-related query assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = pipe(f"{string_dialogue} {prompt_input} Assistant:", 
                    max_length=max_length, 
                    temperature=temperature, 
                    top_p=top_p, 
                    repetition_penalty=1.1, 
                    num_return_sequences=1)[0]['generated_text']
    
    return response

# User-provided prompt
if prompt := st.chat_input("Enter your message:", disabled=not hf_api_token):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
