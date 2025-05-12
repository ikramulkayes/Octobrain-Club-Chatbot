import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import re

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
api_key = os.getenv("API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_api_url = os.getenv("QDRANT_API_URL")

# Configure the generative AI client
genai.configure(api_key=api_key)

# Initialize the Gemini model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

try:
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )
except Exception as e:
    st.error(f"Error initializing Gemini model: {e}")
    model = None

# Initialize Qdrant client for the cloud "clubs" collection
try:
    qdrant_client = QdrantClient(
        url=qdrant_api_url,
        api_key=qdrant_api_key,
    )
except Exception as e:
    st.error(f"Error connecting to Qdrant: {e}")
    qdrant_client = None

# Initialize SentenceTransformer model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}")
    embedding_model = None

# Function to clean HTML tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def search_clubs(query_text, limit=3):
    """
    Given a query about user interests,
    generate an embedding and search the "clubs" collection in Qdrant.
    """
    if not embedding_model or not qdrant_client:
        return []

    try:
        query_embedding = embedding_model.encode(query_text)

        search_results = qdrant_client.search(
            collection_name="clubs",
            query_vector=query_embedding.tolist(),
            limit=limit
        )

        if search_results:
            return [result.payload for result in search_results if result.score >= 0.3]
        return []
    except Exception as e:
        st.error(f"Error searching clubs: {e}")
        return []

def is_interest_query(user_input):
    """
    Returns True if the user input contains interest-related keywords or questions about clubs
    """
    interest_keywords = [
        'like', 'enjoy', 'interest', 'passion', 'hobby', 'love', 'prefer',
        'club', 'join', 'recommend', 'suggest', 'activities', 'sports',
        'coding', 'robotics', 'art', 'music', 'science', 'technology',
        'engineering', 'math', 'stem', 'culture', 'social', 'volunteer',
        'leadership', 'fitness', 'photography'
    ]
    return any(word in user_input.lower() for word in interest_keywords)

def extract_interests(user_input):
    """
    Extract potential interests from user input
    """
    common_interests = [
        "science", "technology", "engineering", "math", "coding", "programming",
        "robotics", "sports", "art", "music", "dance", "photography", "culture",
        "social", "volunteer", "leadership", "fitness", "stem"
    ]

    mentioned_interests = []
    for interest in common_interests:
        if re.search(rf'\b{interest}\b', user_input, re.IGNORECASE):
            mentioned_interests.append(interest)

    return mentioned_interests

def is_followup_query(user_input):
    """
    Returns True if the user input appears to be a follow-up question asking for more recommendations
    """
    followup_keywords = ["more", "another", "different", "else", "other", "additional", "also", "too"]
    return any(word in user_input.lower() for word in followup_keywords)

def generate_club_recommendations(user_interests, clubs):
    """
    Generate club recommendations using the Gemini model
    """
    if not model:
        return "Sorry, I couldn't generate recommendations due to an error with the AI model."

    try:
        clubs_text = "\n\n".join(
            f"Club Name: {club.get('club_name', 'Unknown')}\n"
            f"Motto: {club.get('motto', '')}\n"
            f"Description: {clean_html(club.get('description', ''))}\n"
            f"Tags: {', '.join(club.get('tags', []))}"
            for club in clubs
        )

        prompt = f"""
You are a helpful club advisor for students. Based on the user's interests and the recommended clubs from our database, suggest the top 3 clubs that would be the best fit.

User's Interests:
{user_interests}

Recommended Clubs:
{clubs_text}

For each recommended club, explain why it would be a good match for the user based on their interests. Highlight the key activities and benefits of joining each club. Limit your response to the top 3 clubs only.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return "Sorry, I couldn't generate recommendations at this time. Please try again later."

def generate_general_response(conversation_history, latest_input):
    """
    Generate a general response using the Gemini model
    """
    if not model:
        return "Sorry, I couldn't generate a response due to an error with the AI model."

    try:
        prompt = f"""
You are a helpful assistant specializing in recommending student clubs. Based on the conversation history below and the user's latest input, generate a thoughtful response.

Conversation History:
{conversation_history}

User's Latest Input:
{latest_input}

Response:
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time. Please try again later."

# Streamlit app layout
st.title("Club Recommendation Chatbot")

# Initialize session state to store chat history and detected interests
if "messages" not in st.session_state:
    st.session_state.messages = []
    # System message for context
    system_msg = {"role": "system", "content": "You are a helpful assistant specializing in recommending student clubs."}
    st.session_state.messages.append(system_msg)

    initial_greeting = ("Hello! I'm your club recommendation assistant. Tell me about your interests and activities you enjoy, "
                        "and I'll suggest some clubs that might be a good fit for you!")
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

if "detected_interests" not in st.session_state:
    st.session_state.detected_interests = []

if "last_recommended_clubs" not in st.session_state:
    st.session_state.last_recommended_clubs = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Tell me about your interests..."):
    # Append the user prompt to session state as a dict
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize a variable for response_text
    response_text = ""

    # Extract any new interests from the prompt
    new_interests = extract_interests(prompt)
    if new_interests:
        for interest in new_interests:
            if interest not in st.session_state.detected_interests:
                st.session_state.detected_interests.append(interest)

    # Decide on action based on input
    if is_interest_query(prompt) or is_followup_query(prompt) or st.session_state.detected_interests:
        # Build search query using all detected interests
        search_query = prompt
        if st.session_state.detected_interests:
            interest_str = ", ".join(st.session_state.detected_interests)
            search_query = f"I'm interested in {interest_str}. {prompt}"

        # Search for clubs
        clubs = search_clubs(search_query)

        # Filter out clubs that were recommended in the last message
        if is_followup_query(prompt) and st.session_state.last_recommended_clubs:
            last_club_names = [club.get('club_name') for club in st.session_state.last_recommended_clubs]
            clubs = [club for club in clubs if club.get('club_name') not in last_club_names]

        if clubs:
            # Store these clubs as the last recommended ones
            st.session_state.last_recommended_clubs = clubs

            # Generate recommendations
            response_text = generate_club_recommendations(
                f"{prompt} (Detected interests: {', '.join(st.session_state.detected_interests)})",
                clubs
            )
        else:
            response_text = "I couldn't find any more clubs that match your interests. Would you like to explore clubs in a different area?"
    else:
        # Otherwise, respond based on conversation history
        # Build a conversation history string from the messages (excluding the system message)
        conversation_history = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != "system"
        )
        # Generate a general response
        response_text = generate_general_response(conversation_history, prompt)

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Append the assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Create a sample output file
with open('club_recommendation_chatbot_cloud.py', 'w') as f:
    f.write("""
# Club Recommendation Chatbot (Cloud Version)
# This file contains the code for a Streamlit-based chatbot that recommends clubs based on user interests
# Uses Qdrant Cloud for vector search
# Run with: streamlit run club_recommendation_chatbot_cloud.py --server.enableCORS=false
""")
