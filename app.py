import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_Message(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)

    y = []
    for i in Message:
        if i.isalnum():
            y.append(i)

    Message = y[:]  # here we colone the text
    y.clear()

    # here we remove the stopwords and punctuation
    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
        y.append(ps.stem(i))  # here we are doing stemming

    return " ".join(y)

tfidf = pickle.load(open('C:\\Users\\Akanshu\\AppData\\Roaming\\JetBrains\\PyCharmCE2024.3\\scratches\\vectorizer.pkl', 'rb'))
model = pickle.load(open('C:\\Users\\Akanshu\\AppData\\Roaming\\JetBrains\\PyCharmCE2024.3\\scratches\\model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_Message(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
