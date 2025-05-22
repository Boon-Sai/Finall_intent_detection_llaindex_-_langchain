# Intent Detection RAG System

This is a Retrieval-Augmented Generation (RAG) system for intent detection using LlamaIndex, LangChain, and Streamlit. The system uses ChromaDB for vector storage and Groq's LLM for inference.

## Prerequisites

- Python 3.10
- Conda (for environment management)
- Groq API key

## Setup

1. Create a new conda environment:
```bash
conda create -n intent_detector python=3.10
conda activate intent_detector
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Running the Application

1. Make sure your conda environment is activated:
```bash
conda activate intent_detector
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## How it Works

1. The system loads the dataset from `cleaned_output.csv` and processes it into individual documents.
2. Each document is embedded using the sentence-transformers model and stored in ChromaDB.
3. When a query is entered:
   - The system retrieves relevant context from the vector store
   - Uses the Groq LLM to determine the intent and generate a response
   - Displays the results in the Streamlit interface

## Dataset Structure

The system expects a CSV file (`cleaned_output.csv`) with the following columns:
- query: The input query
- intent: The detected intent
- response: The corresponding response

## Features

- Real-time intent detection
- RAG-based response generation
- User-friendly Streamlit interface
- Efficient vector storage with ChromaDB
- Fine-tuned responses using LangChain 