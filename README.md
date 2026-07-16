# RAG Unsaidtalks FAQ RAG Chatbot

## UnsaidTalks Q&A: AI-Powered Question Answering System Using Google Gemini, LangChain, and FAISS

This is an end-to-end Retrieval-Augmented Generation (RAG) project built using **Google Gemini**, **LangChain**, **FAISS**, and **Streamlit**.

The project is designed for **UnsaidTalks**, a mentorship and placement preparation platform that helps engineering students prepare for software engineering careers through structured courses, mentorship, interview preparation, resume reviews, mock interviews, and placement guidance.

Thousands of students have common questions related to courses, placements, interviews, projects, resume building, and career guidance. Instead of relying entirely on mentors and support staff, this system allows students to ask questions through an AI-powered chatbot and receive accurate answers within seconds using a knowledge base built from the UnsaidTalks FAQ dataset.

---

# Project Highlights

* Uses a real FAQ-style CSV dataset for building the knowledge base.
* Implements Retrieval-Augmented Generation (RAG) for accurate responses.
* Generates vector embeddings from the FAQ dataset.
* Stores embeddings locally using FAISS.
* Retrieves the most relevant FAQ entries before generating responses.
* Provides a clean Streamlit-based chat interface.
* Reduces the workload of mentors and support teams by automating frequently asked questions.
* Easily extendable with new FAQs without retraining the model.
* Accepts PDF and CSV uploads at runtime and updates the FAISS knowledge base immediately.
* Preserves conversation history for more contextual multi-turn answers.

---

# Technologies Used

* Google Gemini API
* LangChain
* Streamlit
* HuggingFace Instructor Embeddings
* FAISS Vector Database
* Python
* Pandas
* dotenv

---

# Installation

## 1. Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Create a `.env` file

Add your Google AI API key.

```env
GOOGLE_API_KEY="your_google_api_key"
```

---

# Usage

Run the Streamlit application

```bash
streamlit run main.py
```

The application will automatically open in your browser.

---

# Creating the Knowledge Base

1. Place the FAQ CSV file inside the project directory.
2. Click the **Create Knowledge Base** button.
3. The application will:

   * Load the FAQ dataset
   * Generate embeddings
   * Store them inside a local FAISS vector database
4. Once complete, a new folder named

```
faiss_index/
```

will be created.

The chatbot is now ready to answer questions.
You can also upload additional PDF or CSV files from the sidebar and index them without restarting the app.

---

# Sample Questions

* What is UnsaidTalks?
* Which courses are offered by UnsaidTalks?
* How can I prepare for software placements?
* Does UnsaidTalks provide mock interviews?
* How do I improve my resume?
* What projects should I build before placements?
* How should I prepare for DSA interviews?
* What are the prerequisites for joining?
* How can I crack product-based company interviews?
* What placement roadmap should I follow in my final year?

---

# Project Structure

```
RAG_Unsaidtalks_FAQ/
│
├── main.py
├── langchain_helper.py
├── create_vector_db.py
├── requirements.txt
├── .env
├── faiss_index/
├── FAQs.csv
├── README.md
└── images/
```

---

# File Description

### `main.py`

The main Streamlit application responsible for the chatbot interface and user interactions.

### `langchain_helper.py`

Contains all LangChain logic, including loading the vector database, retrieval pipeline, prompt template, and interaction with Google Gemini.

### `create_vector_db.py`

Reads the FAQ dataset, generates embeddings using HuggingFace Instructor Embeddings, and creates the FAISS vector database.

### `FAQs.csv`

Contains the prompt-response dataset used to build the knowledge base.

### `requirements.txt`

Lists all Python dependencies required to run the project.

### `.env`

Stores environment variables such as the Google Gemini API key.

### `faiss_index/`

Contains the locally generated FAISS vector database used for semantic search.

---

# How It Works

1. Load the FAQ CSV dataset.
2. Convert each FAQ into vector embeddings.
3. Store embeddings in a FAISS vector database.
4. Accept user questions through Streamlit.
5. Retrieve the most relevant FAQ entries.
6. Pass the retrieved context to Google Gemini using LangChain.
7. Generate an accurate and context-aware answer.
8. Display the response in a conversational chat interface.

---

# Future Improvements

* Voice-based interaction
* Mentor escalation for unanswered questions
* Hybrid search (keyword + semantic)
* ChromaDB or Pinecone integration
* Admin dashboard for managing FAQs
* Source citations for retrieved answers
* Multi-language support

---

# License

This project is intended for educational and internal use. Feel free to modify and extend it according to your organization's requirements.
