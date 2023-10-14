import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import time

# Define a function to get the most similar response
def get_most_similar_response(df, query, top_k=1):
    vectorizer = TfidfVectorizer()
    all_data = list(df['Question']) + [query]
    tfidf_matrix = vectorizer.fit_transform(all_data)
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)
    sorted_indexes = similarity_scores.argsort()[0][::-1][:top_k]
    most_similar_responses = df.iloc[sorted_indexes]['Answer'].values
    return most_similar_responses

def is_insufficient(prompt):
    return len(prompt.split()) <= 1

# Sample DataFrame for your basketball QA chatbot
df_basketball = pd.read_csv('basketball_qa.csv')

st.title("Basketball Q&A Chatbot")

description = """
This chatbot is your ultimate companion for exploring the exciting world of "Basketball." Whether you have questions about rules, famous players, or the history of the game, I'm here to help! Please feel free to ask any basketball-related questions, and I'll provide you with the best answers.
"""

st.markdown(description)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.text_input("You:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if is_insufficient(prompt):
        insufficient_response = "Insufficient Prompt. Please clarify what you want to know."
        with st.chat_message("assistant"):
            st.markdown(insufficient_response)
        st.session_state.messages.append({"role": "assistant", "content": insufficient_response, "related_query": prompt})
    else:
        # Check if the same prompt was already answered previously
        previous_responses = [m["content"] for m in st.session_state.messages if m["role"] == "assistant" and m["related_query"] == prompt]

        if previous_responses:
            for response in previous_responses:
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            # Get and display assistant response in chat message container
            responses_basketball = get_most_similar_response(df_basketball, prompt)

            if responses_basketball.any():
                for response in responses_basketball:
                    with st.chat_message("assistant"):
                        st.markdown(f"{response}")
                    st.session_state.messages.append({"role": "assistant", "content": f"{response}", "related_query": prompt})
            else:
                not_understood_response = "I'm sorry, I couldn't find an answer to your question."
                with st.chat_message("assistant"):
                    st.markdown(not_understood_response)
                st.session_state.messages.append({"role": "assistant", "content": not_understood_response, "related_query": prompt})
