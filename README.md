# 📖 Bhagavad Gita Chatbot

A conversational AI chatbot that answers questions about the **Bhagavad Gita** using **Retrieval-Augmented Generation (RAG)**.  
It provides Sanskrit verses, translations, and purports as context, ensuring answers are grounded in scripture.  

Users can create or enter a **User ID** to save and retrieve their chat history across sessions.  

---

## ✨ Features

- 💬 **Ask Questions Freely** – Interact without needing login/signup.
- 📂 **Persistent Chat History** – Save chats by entering your unique User ID.
- 🔍 **RAG Pipeline** – Answers grounded in verses, translations, and purports from the Bhagavad Gita.
- 📖 **Verse Context** – Sanskrit, translation, and purport provided for each answer.
- 🧠 **Multi-Query Retrieval** – Expands user queries into multiple forms to fetch the best context.
- ⚡ **Streamed Responses** – AI responses stream into the chat for a natural feel.
- 🗑️ **Session Management** – Create, rename, and delete chat sessions in the sidebar.

---

## 🧩 How RAG Works in This Project

1. **Vector Database (ChromaDB)**  
   - Verses, translations, and purports are embedded with `GoogleGenerativeAIEmbeddings`.  
   - Stored in a local Chroma vectorstore for semantic retrieval.  

2. **Retriever**  
   - Uses **MMR (Maximal Marginal Relevance)** and **Multi-Query Retrieval** to fetch diverse and relevant passages.  

3. **LLM (Language Model)**  
   - `ChatGroq (Llama 3.1 8B Instant)` (or Google Gemini if enabled).  
   - Takes user query + retrieved context and generates a grounded answer.  

4. **Answer Generation**  
   - Always includes the **original Sanskrit verse**.  
   - Provides translation and concise purport (100–200 words).  
   - Stays strictly within the Bhagavad Gita knowledge base.  

---

## 🚀 Deployment

This project is built with **Streamlit** and you can access it from the following link

Try it now! - https://the-gita-gpt.streamlit.app/