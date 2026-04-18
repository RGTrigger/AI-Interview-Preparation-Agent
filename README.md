# 🤖 AI Interview Preparation Agent

An advanced **Agentic AI-powered interview assistant** built using **LangGraph, RAG (Retrieval-Augmented Generation), and Streamlit**, designed to help students prepare for technical interviews with **real-time feedback, memory-aware conversations, and structured evaluation**.

---

## 🚀 Live Demo

👉 *(Add after deployment)*
https://your-app.streamlit.app

---

## 📸 Application Screenshots

### 🏠 Home Interface

![Home UI](assets/home.png)

### 💬 Chat Interface

![Chat UI](assets/chat.png)

### 🎯 Interview Mode

![Interview Mode](assets/interview.png)

> 📌 Create an `assets/` folder and add screenshots here.

---

## 📌 Project Overview

This project is a **complete Agentic AI system** that simulates a real interview assistant.

It can:

* Answer technical questions using a **knowledge base (RAG)**
* Conduct **mock interviews**
* Evaluate answers with scoring
* Maintain **context across multiple interactions**
* Provide topic-based insights

---

## 🧠 Key Features

* 🔍 RAG-based answering (ChromaDB)
* 🧩 LangGraph workflow (multi-node architecture)
* 💬 Memory-enabled chat (thread_id)
* 🎯 Interview mode with scoring
* 📊 Faithfulness evaluation + retry logic
* 🛠️ Tool routing support
* 🎨 Custom Streamlit UI

---

## 🏗️ System Architecture

```
User Input
   ↓
Memory Node
   ↓
Router Node
   ↓
 ┌───────────────┬───────────────┬───────────────┐
 ↓               ↓               ↓
Retrieval Node   Tool Node       Skip Node
 ↓               ↓               ↓
        Answer Node
             ↓
        Evaluation Node
             ↓
          Save Node
             ↓
            END
```

---

## ⚙️ Tech Stack

| Category   | Technology           |
| ---------- | -------------------- |
| Frontend   | Streamlit            |
| Backend    | Python               |
| LLM        | Groq                 |
| Framework  | LangGraph            |
| Vector DB  | ChromaDB             |
| Embeddings | SentenceTransformers |
| Memory     | MemorySaver          |
| Database   | SQLite               |

---

## 📂 Project Structure

```
AI-Interview-Preparation-Agent/
│
├── capstone_streamlit.py
├── capstone_streamlit.py.ipynb
├── chat_history.db
├── requirements.txt
├── README.md
└── .env (not included)
```

---

# ⚙️ Setup

## 1. Clone the repository

```
git clone https://github.com/RGTrigger/AI-Interview-Preparation-Agent.git
cd "AI-Interview-Preparation-Agent"
```

## 2. Create and activate a virtual environment

```
python -m venv .venv
```

### Windows (PowerShell)

```
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```
source .venv/bin/activate
```

---

## 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

# ▶️ Run Locally

```
streamlit run capstone_streamlit.py
```

✔ The app will:

* Launch locally
* Start each session with a new chat automatically

---

# 🚀 Deployment

## Streamlit Community Cloud

### Recommended Steps:

1. Push the project to GitHub
2. Add secret:

   ```
   GROQ_API_KEY
   ```
3. Deploy from GitHub repo

### Entry Point:

```
capstone_streamlit.py
```

---

## ⚠️ Deployment Notes

* Paths are handled relative to script (deployment-safe)
* Do NOT upload `.env`
* `chat_history.db` is runtime-only
* `chat_history.json` is NOT used

---

# 🧹 Recommended Git Hygiene

Add these to `.gitignore`:

```
.env
__pycache__/
chat_history.db
```

---

## ⚡ How It Works

1. User inputs query
2. Memory stores conversation
3. Router decides flow
4. Retrieval / Tool execution
5. LLM generates answer
6. Evaluation checks faithfulness
7. Final response displayed

---

## 🧪 Testing

* Retrieval testing
* Memory testing
* Red-team testing
* Faithfulness scoring
* Interview evaluation

---

## 📊 Unique Highlights

* Agentic AI architecture
* Self-correcting responses
* Interview scoring system
* Clean UI
* Context-aware intelligence

---

## 🔮 Future Improvements

* Export chat sessions
* User authentication
* Topic analytics dashboard
* Prompt customization
* Cloud-based storage

---

## 👨‍💻 Developer

**Gaurav (RGtrigger)**
B.Tech – CSCE
KIIT University

---

## 📄 License

Academic / Educational Use Only

---

## ⭐ Final Note

This project demonstrates:

* Agentic AI systems
* RAG pipelines
* Real-time AI applications with memory

---

Thank you.

Regards,
Gaurav
