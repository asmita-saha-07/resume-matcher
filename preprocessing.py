import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    text=text.lower()
    #remove url
    text=re.sub(r'https?://\S+|www\.\S+', '', text)
    #remove email
    text=re.sub(r'\S+@\S+', '', text)
    #remove ph no.
    text=re.sub(r'\+?\d[\d\s().-]{7,}\d', '', text)
    #remove special characters
    text=re.sub(r'[^a-z0-9\s]', ' ', text)
    #remove extra spaces
    text=re.sub(r'\s+', ' ', text).strip()
    words=text.split()
    words=[word for word in words if word not in ENGLISH_STOP_WORDS]
    final_text=" ".join(words)

    return final_text
