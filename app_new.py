import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging

# Set logging level to avoid deprecation warning messages
logging.set_verbosity_error()

st.title("Finance Chatbot")

@st.cache_resource()
def load_model_and_tokenizer():
    model_name = "tito92/finance_finetune_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=100):
    # Format the input prompt as specified
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")  
    return result[0]['generated_text']

model, tokenizer = load_model_and_tokenizer()
user_input = st.text_input("Enter your query:")

if user_input:
    response = generate_text(user_input, model, tokenizer)
    st.write(response)
