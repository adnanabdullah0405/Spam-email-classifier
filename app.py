import streamlit as st
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords

# Load the saved model and vectorizer
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Download stopwords if not already downloaded
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    return text

# Streamlit UI
st.set_page_config(page_title="Spam Email Detection", page_icon="🚫", layout="wide")

# Add a header with title and style
st.markdown("""
    <h1 style='text-align: center; color: #2e3b4e;'>🚫 Welcome to DK's Spam Email Detection System 🚫</h1>
    <p style='text-align: center; color: #4f4f4f; font-size: 20px;'>Enter an email and find out whether it is <strong>Spam</strong> or <strong>Ham</strong>.</p>
    <style>
    .stButton>button {
        background-color: #FF5733;
        color: white;
        font-size: 18px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #C70039;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Input section in the main area
st.markdown("""
    <p style='font-size: 18px; color: #5D5D5D; text-align: center;'>
        Type or paste your email below, and let's check if it's <strong>Spam</strong> or <strong>Ham</strong>! 🚨
    </p>
""", unsafe_allow_html=True)

# Text input for the user email
user_email = st.text_area("Enter your email here", height=150, max_chars=500)

# If the user clicks "Check" button
if st.button("Check Email"):
    if user_email:
        # Preprocess the input text
        cleaned_text = preprocess_text(user_email)
        
        # Convert the input text into TF-IDF features
        text_tfidf = tfidf.transform([cleaned_text]).toarray()
        
        # Predict with the trained model
        prediction = model.predict(text_tfidf)
        
        # Display the result with style
        if prediction[0] == 1:
            st.markdown(f"<h2 style='text-align: center; color: red;'>🚨 **Spam** Email Detected 🚨</h2>", unsafe_allow_html=True)
            st.write("**Warning:** This email is classified as **Spam**! Be cautious and avoid clicking any suspicious links.")
        else:
            st.markdown(f"<h2 style='text-align: center; color: green;'>✅ **Ham** Email Detected ✅</h2>", unsafe_allow_html=True)
            st.write("This email is safe! It's a **Ham** (non-spam) email.")
    else:
        st.warning("Please enter an email to check.")

# Footer Section
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #8a8a8a;'>
        <p>Powered by DK (Muhammad Adnan) | NLP Spam Detection App</p>
        <p><strong>Tip:</strong> Always be cautious with suspicious emails and unknown senders.</p>
    </div>
""", unsafe_allow_html=True)

