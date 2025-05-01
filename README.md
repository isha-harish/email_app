# Gmail AI Inbox Assistant

An intelligent email analysis and summarization tool using the Gmail API, Hugging Face LLMs, and Mistral structured output models—powered by LangChain. This app fetches emails from your Gmail inbox, summarizes and classifies them, performs sentiment analysis, and enables interactive question-answering over your email history using Retrieval-Augmented Generation (RAG) via LangChain and LangGraph.

---

```
## Folder Structure
email_app
│
├── app.py        
│
├── gmail_fetcher.py
├── requirements.txt
├── rag.py
├── Results           
│   └──results(images)
│
├── Templates       
│   └── index.html       
│
└── Readme.md           # Documentation for the project.
```
---
## Steps to Run the App
### 1. Clone the Repository
Add your credentials.json from Gmail API 
```bash
git clone https://github.com/isha-harish/email_app.git
cd email_app
```
### 2. Set Up a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install the Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask App
```bash
python app.py 
```

### 5. Access the App
Open your browser and go to http://127.0.0.1:5000/


### 6. For the chatbot 
Run rag.py and interact with ur AI Assistant regarding emails 
```bash
python rag.py 
```
