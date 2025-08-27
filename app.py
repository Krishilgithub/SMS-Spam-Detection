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

# Add multiple potential paths for NLTK data
additional_paths = [
    '/home/appuser/nltk_data',
    '/home/adminuser/venv/nltk_data',
    '/tmp/nltk_data',
    './nltk_data'
]
for path in additional_paths:
    if path not in nltk.data.path:
        nltk.data.path.append(path)

# Download required NLTK data if not available
@st.cache_data
def download_nltk_data():
    # Download punkt (older versions)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    
    # Download punkt_tab (newer versions)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
        except:
            pass  # If punkt_tab doesn't exist in this version, ignore
    
    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

# Download NLTK data
download_nltk_data()

ps = PorterStemmer()

def transform_text(text):       #* we can also dump the function into pickle and then extract it here
  text = text.lower()
  
  # Try different tokenization methods for compatibility
  try:
    text = nltk.word_tokenize(text)
  except LookupError:
    # Fallback to simple split if NLTK tokenization fails
    text = text.split()

  y = []

  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  # Get stopwords with error handling
  try:
    stop_words = stopwords.words('english')
  except LookupError:
    # Fallback to basic English stopwords if NLTK data not available
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once']

  for i in text:
    if i not in stop_words and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  # Apply stemming with error handling
  for i in text:
    try:
      y.append(ps.stem(i))
    except:
      # If stemming fails, just append the original word
      y.append(i)

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