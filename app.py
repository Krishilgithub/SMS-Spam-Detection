import streamlit as st
import pickle
import nltk   #type: ignore
from nltk.corpus import stopwords   #type: ignore
import string
from nltk.stem import PorterStemmer   #type: ignore

# Configure NLTK data path for deployment
import os
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data if not available
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)

# Download NLTK data
download_nltk_data()

ps = PorterStemmer()

def transform_text(text):       #* we can also dump the function into pickle and then extract it here
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []

  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

# Load the trained model and vectorizer
try:
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found! Please run the notebook to create 'tfidf.pkl' and 'model.pkl'")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("SMS Spam Classifier")
st.write("Enter an SMS message to check if it's spam or not.")

input_sms = st.text_area("Enter the SMS message", placeholder="Type your SMS message here...")

if st.button("Predict"):
  if input_sms.strip():
    try:
      # 1. Preprocess
      transform_sms = transform_text(input_sms)
      
      # 2. Vectorize
      vector_input = tfidf.transform([transform_sms])
      
      # 3. Predict
      result = model.predict(vector_input)[0]
      probability = model.predict_proba(vector_input)[0]
      
      # 4. Display
      if result == 1:
        st.error("**SPAM** detected!")
        st.write(f"Confidence: {probability[1]:.2%}")
      else:
        st.success("**NOT SPAM** - Safe message")
        st.write(f"Confidence: {probability[0]:.2%}")
        
      # Show preprocessing details in expander
      with st.expander("See preprocessing details"):
        st.write(f"**Original message:** {input_sms}")
        st.write(f"**Processed message:** {transform_sms}")
        st.write(f"**Vector shape:** {vector_input.shape}")
        
    except Exception as e:
      st.error(f"Error during prediction: {str(e)}")
  else:
    st.warning("Please enter a message to classify!")