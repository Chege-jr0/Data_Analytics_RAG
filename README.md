#  Data Analytics RAG

Chat with your CSV and Excel files using plain English — powered by FastAPI + LangChain + OpenAI.

---

##  Setup (Step by Step)

### 1. Open this folder in VS Code terminal

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API Key
Create a `.env` file in the project folder:
```
OPENAI_API_KEY=your-openai-api-key-here
```
Or set it directly in the terminal:
```bash
set OPENAI_API_KEY=your-openai-api-key-here   # Windows
```

### 5. Run the app
```bash
uvicorn main:app --reload
```

### 6. Open the interactive API docs
Go to: **http://127.0.0.1:8000/docs**

---

##  How to Use

1. **Upload your CSV/Excel** → `POST /upload`
2. **Ask questions** → `POST /ask`
   - "What are the top 5 values in column X?"
   - "What is the average sales?"
   - "Are there any missing values?"
   - "What trends do you see in the data?"
3. **Check data info** → `GET /data-info`

---

##  Project Structure
```
data-analytics-rag/
├── main.py          # FastAPI app & endpoints
├── ingest.py        # Converts DataFrame to text chunks
├── rag.py           # LangChain RAG chain logic
├── requirements.txt
└── README.md
```

---

##  Example Questions to Ask
- "How many rows are in this dataset?"
- "What is the average of a certain column?"
- "Which category has the highest value?"
- "Are there missing values in the data?"
- "Give me a summary of the dataset"
- "What are the top 5 categorical columns?"
