# 💼 Chat with Sukhdeep Singh – AI Career Agent

## Deployed with Streamlit : https://chat-with-sukhdeep-7qpfowc9luszbvpxhs5y9j.streamlit.app/

This Streamlit app is an intelligent, AI-powered career assistant built to answer questions about **Sukhdeep Singh's** professional background, experience, and projects. It leverages a Retrieval-Augmented Generation (RAG) pipeline with OpenAI and Cassandra to deliver personalized responses using private document embeddings.

---

## 🧠 Features

- 🤖 **Conversational AI** powered by OpenAI (GPT-4o-mini)
- 🔍 **RAG pipeline** using a vector store (Cassandra + OpenAIEmbeddings)
- 📂 **Semantic search** across embedded personal documents
- 📬 **Pushover notifications** for contact or unknown queries
- 📧 **Email capture** via tool usage
- 🧠 **Dynamic memory & tool calling** with OpenAI's function calling API
- 🌐 Built using **Streamlit** for a user-friendly web interface

---

## 🗂️ File Structure

```
.
├── app.py              # Main Streamlit app
├── .env                # Environment file for local development
└── README.md           # Project documentation
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/chat-with-sukhdeep.git
cd chat-with-sukhdeep
```

### 2. Install Dependencies

Make sure you have Python 3.8+ and install required packages:

```bash
pip install -r requirements.txt
```

_(Sample `requirements.txt` should include: `streamlit`, `openai`, `cassio`, `langchain`, `python-dotenv`, `requests`)_

### 3. Configure Environment Variables

You can use either:

#### a. **Streamlit Secrets (recommended for production)**

Create a `.streamlit/secrets.toml` file:

```toml
[openai]
api_key = "sk-..."

[astra_db]
token = "AstraDBToken"
id = "AstraDBID"

[pushover]
user = "PushoverUser"
token = "PushoverToken"
```

#### b. **Local `.env` file (for development)**

```bash
OPENAI_API_KEY=sk-...
ASTRA_DB_APPLICATION_TOKEN=...
ASTRA_DB_ID=...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
```

### 4. Launch the App

```bash
streamlit run app.py
```

---

## 🛠️ Key Components

| Section                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **RAG Pipeline**       | Connects to Cassandra vector store using `OpenAIEmbeddings` for retrieval. |
| **Tool Functions**     | Functions used for searching documents, storing user info, and logging unknowns. |
| **Pushover Alerts**    | Sends push notifications for contact or unknown questions.                 |
| **Agent Loop**         | Orchestrates interaction between user, OpenAI, and tools dynamically.       |
| **Chat UI**            | Simple chat interface built with Streamlit's new `st.chat_*` elements.     |

---

## 🤝 Acknowledgements

- [OpenAI](https://openai.com/)
- [Cassio (DataStax)](https://docs.datastax.com/en/astra/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Pushover](https://pushover.net/)

---

## 📌 Notes

- Ensure your Astra DB table (`career_agent_multi_doc_v2`) is already populated.
- The assistant only responds to queries related to **Sukhdeep Singh’s** professional life.
- Contact details are only logged if the user explicitly shares them.

---

## 📸 Example Use Cases

- “What projects has Sukhdeep worked on?”
- “Tell me about Sukhdeep’s experience with cloud infrastructure.”
- “I’d like to get in touch. My email is john@example.com.”
