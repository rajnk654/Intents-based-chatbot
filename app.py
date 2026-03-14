
import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#eef2f7,#e3ecff);
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1{
    text-align:center;
    color:#1e2a38;
    font-weight:700;
}

/* Welcome banner */
.welcome-banner{
    text-align:center;
    padding:15px;
    background:linear-gradient(90deg,#2c3e50,#4ca1af);
    color:white;
    border-radius:12px;
    margin-bottom:20px;
    font-size:16px;
    box-shadow:0 4px 10px rgba(0,0,0,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1f2a40,#141b2d);
}

/* Sidebar text */
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Input box */
.stTextInput input{
    border-radius:25px;
    border:1px solid #ccc;
    padding:12px 18px;
    font-size:15px;
    background:#ffffff;
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
}

/* User bubble */
.user-bubble{
    background:linear-gradient(135deg,#0084ff,#00c6ff);
    color:white;
    padding:14px 18px;
    border-radius:20px 20px 5px 20px;
    margin:12px 0;
    width:fit-content;
    max-width:70%;
    margin-left:auto;
    font-size:15px;
    box-shadow:0 3px 8px rgba(0,0,0,0.2);
}

/* Bot bubble */
.bot-bubble{
    background:white;
    color:#333;
    padding:14px 18px;
    border-radius:20px 20px 20px 5px;
    margin:12px 0;
    width:fit-content;
    max-width:70%;
    font-size:15px;
    box-shadow:0 3px 8px rgba(0,0,0,0.15);
}

/* Divider */
hr{
    border:none;
    height:1px;
    background:#ddd;
    margin:20px 0;
}

</style>
""", unsafe_allow_html=True)


# ---------------- NLTK SETUP ---------------- #

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')


# ---------------- LOAD INTENTS ---------------- #

file_path = "intents.json"

with open(file_path) as file:
    intents = json.load(file)


# ---------------- TRAIN MODEL ---------------- #

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []

for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

x = vectorizer.fit_transform(patterns)
y = tags

clf.fit(x, y)


# ---------------- CHATBOT FUNCTION ---------------- #

def chatbot(user_input):

    input_data = vectorizer.transform([user_input])
    tag = clf.predict(input_data)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])


counter = 0


# ---------------- MAIN APP ---------------- #

def main():

    global counter

    st.markdown("<h1>🤖 AI Intent Based Chatbot</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-banner">
    Welcome! Ask me anything related to the trained intents.
    </div>
    """, unsafe_allow_html=True)

    menu = ["Home", "Conversation History", "About"]

    choice = st.sidebar.selectbox("Menu", menu)


    # Create chat log file
    if not os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["User Input", "Chatbot Response", "Timestamp"])


    # ---------------- HOME ---------------- #

    if choice == "Home":

        st.write("Type your message below to chat with the bot.")

        counter += 1

        user_input = st.text_input("You:", key=str(counter))

        if user_input:

            response = chatbot(user_input)

            st.markdown(
                f'<div class="user-bubble">You: {user_input}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="bot-bubble">Bot: {response}</div>',
                unsafe_allow_html=True
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("chat_log.csv", "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([user_input, response, timestamp])

            if response.lower() in ["bye", "goodbye"]:
                st.write("Thank you for chatting! 👋")
                st.stop()


    # ---------------- HISTORY ---------------- #

    elif choice == "Conversation History":

        st.header("Conversation History")

        if os.path.exists("chat_log.csv"):

            with open("chat_log.csv", "r", encoding="utf-8") as file:

                reader = csv.reader(file)
                next(reader)

                for row in reader:

                    st.markdown(
                        f'<div class="user-bubble">User: {row[0]}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="bot-bubble">Bot: {row[1]}</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(row[2])

                    st.markdown("---")

        else:
            st.write("No conversation history available.")



    # ---------------- ABOUT ---------------- #

    elif choice == "About":

        st.subheader("About This Chatbot")

        st.write("""
This chatbot is built using **Natural Language Processing (NLP)**.

### Techniques Used
- TF-IDF Vectorization
- Logistic Regression Classification
- Intent Based Responses

### Technologies
- Python
- NLTK
- Scikit-learn
- Streamlit
        """)


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    main()