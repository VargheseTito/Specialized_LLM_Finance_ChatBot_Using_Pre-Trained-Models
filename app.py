import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from transformers.utils import logging

# Set logging level to avoid deprecation warning messages
logging.set_verbosity_error()


#def load_model_and_tokenizer():
    #model_name = "tito92/finance_finetune_model"
    #tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    #model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)
    #return model, tokenizer
st.title("Finance Chatbot")
user_input = st.text_input("Enter your query:")

@st.cache_resource
def generate_text(prompt, max_length=100):
    # Format the input prompt as specified
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
    result = pipe(f"<s>[INST] {prompt} [/INST]")  
    return result[0]['generated_text']



if user_input:
    model_name = "tito92/finance_finetune_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    response = generate_text(user_input)
    st.write(response)


