import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import random

# Function to get the most similar response
def get_most_similar_response(df, query, top_k=1):
    vectorizer = TfidfVectorizer()
    all_data = list(df['Question']) + [query]
    tfidf_matrix = vectorizer.fit_transform(all_data)
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)
    sorted_indexes = similarity_scores.argsort()[0][::-1][:top_k]
    most_similar_responses = df.iloc[sorted_indexes]
    return most_similar_responses

# Read the data from the CSV file
df = pd.read_csv('basketball_qa.csv')

# Set the title and description for the Streamlit app
st.title("Basketball Q&A Chatbot")
st.markdown("""
This chatbot is your ultimate companion for exploring the exciting world of "Basketball." It specializes in providing answers to basketball-related questions. Whether you're a die-hard fan or just curious about the sport, feel free to ask questions, and the chatbot will provide you with informative answers.

To interact with the chatbot, simply type your question in the input box below, and the chatbot will respond accordingly. For example, you can ask questions about basketball rules, famous players, or historical events.

Give it a try, and let's dive into the world of basketball!
""")

# Initialize empty lists to store chatbot responses and questions
chatbot_responses = []
chatbot_questions = []

# Initialize a variable to track the last user interaction time
last_interaction_time = time.time()

while True:
    current_time = time.time()
    # Check if it has been at least 10 seconds since the last user interaction
    if current_time - last_interaction_time >= 10:
        # Randomly select a question from the dataset
        random_index = random.randint(0, len(df) - 1)
        random_question = df.iloc[random_index]['Question']

        chatbot_questions.append(random_question)
        chatbot_responses.append("I'm here to help with any basketball-related questions. Feel free to ask!")

        st.text(f"Chatbot: {random_question}")
        last_interaction_time = current_time

    query = st.text_input("You: ")
    last_interaction_time = time.time()  # Update the last interaction time with user input

    if query.lower() == 'exit':
        # If the user types 'exit,' end the conversation
        chatbot_questions.append("exit")
        chatbot_responses.append("Chatbot: Goodbye!")

        # Create a DataFrame for chatbot responses and questions
        chatbot_df = pd.DataFrame({
            'Question': chatbot_questions,
            'Answer': chatbot_responses
        })

        # Save the chatbot responses to a CSV file
        chatbot_df.to_csv('chatbot_responses.csv', index=False)

        st.text("Chatbot: Goodbye!")
        break

    response_df = get_most_similar_response(df, query)
    if not response_df.empty:
        response = response_df.iloc[0]['Answer']
        question = response_df.iloc[0]['Question']

        chatbot_questions.append(question)
        chatbot_responses.append(response)

        st.text(f"Chatbot: {response}")
