# app.py
import streamlit as st
import os
import json
import uuid
import requests
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
import cassio

# --- 1. Configuration and Setup ---

# Agent's persona
AGENT_NAME = "Sukhdeep Singh"


try:  
    # Production: Load from Streamlit secrets
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
    ASTRA_DB_TOKEN = st.secrets["astra_db"]["token"]
    ASTRA_DB_ID = st.secrets["astra_db"]["id"]
    PUSHOVER_USER = st.secrets["pushover"]["user"]
    PUSHOVER_TOKEN = st.secrets["pushover"]["token"]
    print("Successfully loaded credentials from Streamlit secrets.")

except (FileNotFoundError, KeyError) as e:
    # Local Development: Fallback to .env file
    print(f"Could not find Streamlit secrets, falling back to .env file. Error: {e}")
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    PUSHOVER_USER = os.getenv("PUSHOVER_USER")
    PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
    print("Loaded credentials from local .env file.")


# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)


# --- 2. RAG Pipeline (Vector Database) ---

@st.cache_resource
def configure_rag_pipeline():
    """
    Sets up the connection to Astra DB and returns a retriever object.
    Uses Streamlit's cache to prevent re-initializing on every script run.
    """
    print("Initializing RAG pipeline...")
    cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Cassandra(
        embedding=embeddings,
        table_name="career_agent_multi_doc_v2",
        session=None,
        keyspace=None
    )
    print("RAG pipeline initialized successfully.")
    # We assume the vector store is populated by a separate ingestion script.
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = configure_rag_pipeline()


# --- 3. Tool Functions ---

def search_career_info(query: str):
    """Searches my personal documents for information related to the user's query."""
    print(f"Searching my documents for: '{query}'...")
    try:
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return {"context": context or "NO_INFORMATION_FOUND"}
    except Exception as e:
        print(f"Error during RAG search: {e}")
        return {"context": "Error searching for information. Please try again."}


def push(message: str):
    """Sends a push notification via Pushover."""
    print(f"Pushover Notification: {message}")
    # The actual API call is here for production use.
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_TOKEN,
            "user": PUSHOVER_USER,
            "message": message
        }, timeout=5)
    except requests.RequestException as e:
        print(f"Failed to send push notification: {e}")


def record_user_details(email: str, name="Not provided", notes="Not provided"):
    """Makes a note of a user's contact details."""
    notification_message = f"New contact from my personal site: {name} ({email}). Notes: {notes}"
    push(notification_message)
    return {"status": "success", "message": f"Great, I've got your details: {email}. I'll be in touch soon. Thanks!"}


def record_unknown_question(question: str):
    """Makes a note of a question I couldn't answer."""
    push(f"On my personal site, I couldn't answer: '{question}'")
    return {"status": "logged", "response_to_user": "That's a great question. I don't have the specifics in my documents, but I've made a note to look into it and will update my knowledge base."}


# --- 4. Tool Definitions & System Prompt ---

tools = [
    {"type": "function", "function": {"name": "search_career_info", "description": "Searches my personal documents (CV, project descriptions, skill summaries) to answer questions about my career, experience, skills, and projects. Use this for any question about my professional life.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The user's question, optimized for semantic search."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "record_user_details", "description": "Saves a user's contact details if they explicitly ask to get in touch or leave their information. Only call this if a user provides an email address.", "parameters": {"type": "object", "properties": {"email": {"type": "string", "description": "The user's email address."}, "name": {"type": "string", "description": "The user's name, if provided."}, "notes": {"type": "string", "description": "Any additional notes or the context of the user's request."}}, "required": ["email"]}}},
    {"type": "function", "function": {"name": "record_unknown_question", "description": "Call this tool if the 'search_career_info' tool returns 'NO_INFORMATION_FOUND'. It logs the user's original question so I can improve my knowledge base. It also provides a polite acknowledgement to the user.", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The original question that could not be answered."}}, "required": ["question"]}}},
]

AVAILABLE_TOOLS = {
    "search_career_info": search_career_info,
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}

system_prompt = f"""You are a personal AI agent representing {AGENT_NAME}.
Your name IS {AGENT_NAME}. You are not an "assistant" or a "chatbot"; you are the digital embodiment of {AGENT_NAME}.
- Speak in the first person ("I", "me", "my").
- Your tone should be professional, friendly, and confident.
- Your primary goal is to answer questions about my professional life using the `search_career_info` tool. This is your main function.
- If a user asks a question and the `search_career_info` tool returns 'NO_INFORMATION_FOUND', you MUST use the `record_unknown_question` tool. This is very important. Do not invent answers.
- Only use the `record_user_details` tool if a user *explicitly* provides their contact information and asks to be contacted. Do not ask for their details.
- If the user asks a general knowledge question or something unrelated to my career, politely decline and steer the conversation back to your purpose. Example: "My purpose is to answer questions about my professional background. I don't have information on that topic, but I'd be happy to discuss my projects or skills."
- Keep your answers concise and to the point.
"""

# --- 5. Agent Logic ---

def get_agent_response(chat_history: list) -> str:
    """
    Manages the agent interaction loop: calls the OpenAI API,
    executes tools if needed, and returns the final response.
    """
    api_messages = [{"role": "system", "content": system_prompt}] + chat_history
    
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                tools=tools,
                tool_choice="auto"
            )
            response_message = response.choices[0].message

            if not response_message.tool_calls:
                return response_message.content

            api_messages.append(response_message)
            tool_results = []
            for tc in response_message.tool_calls:
                function_to_call = AVAILABLE_TOOLS.get(tc.function.name)
                if not function_to_call:
                    tool_results.append({"tool_call_id": tc.id, "role": "tool", "name": tc.function.name, "content": json.dumps({"error": f"Tool '{tc.function.name}' not found."})})
                    continue

                function_args = json.loads(tc.function.arguments)
                function_response = function_to_call(**function_args)
                tool_results.append({"tool_call_id": tc.id, "role": "tool", "name": tc.function.name, "content": json.dumps(function_response)})
            
            api_messages.extend(tool_results)

        except Exception as e:
            st.error(f"An error occurred with the OpenAI API: {e}")
            return "Sorry, I encountered an error. Please try again."


# --- 6. Streamlit Application UI ---

st.set_page_config(
    page_title=f"Chat with {AGENT_NAME}",
    page_icon="🤖",
    layout="centered"
)
st.title(f"Chat with {AGENT_NAME}")

# Initialize chat history in session state if it's not already there.
# This list will persist for the duration of the browser session.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display introduction and chat history
st.markdown("Hello! I'm a conversational agent built to answer questions about Sukhdeep's professional background. Feel free to ask me about his career, skills, or projects.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and agent interaction
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # The agent now gets the history directly from the session state
            final_response = get_agent_response(st.session_state.messages)
            message_placeholder.markdown(final_response)
    
    # The new response is appended, and the whole history is preserved in the session
    st.session_state.messages.append({"role": "assistant", "content": final_response})
