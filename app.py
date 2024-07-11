import streamlit as st
import requests
import json

# Title of the Streamlit app
st.title("Fine Tuned Finance Chat Bot")

# Input field for ngrok URL
ngrok_url = st.text_input("Enter the ngrok URL:", value="http://localhost:5000")

# Input field for the prompt
prompt = st.text_area("Enter your prompt:")

# Button to generate text
if st.button("Generate Text"):
    # Check if ngrok URL and prompt are provided
    if ngrok_url and prompt:
        # Define the function to generate response
        def generate_response(prompt):
            headers = {'Content-Type': 'application/json'}
            payload = {'prompt': prompt}
            response = requests.post(f"{ngrok_url}/generate", headers=headers, data=json.dumps(payload))
            
            try:
                response_json = response.json()
                return response_json['generated_text']
            except json.JSONDecodeError:
                st.error("Error decoding JSON response")
                st.write("Response text:", response.text)
                return None
        
        # Generate response
        response = generate_response(prompt)
        
        # Display the response
        if response:
            st.subheader("Generated Text:")
            st.write(response)
    else:
        st.error("Please enter both ngrok URL and prompt")
