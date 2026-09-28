# MP1 Prompt Lab -- Compare LLM Strategies 

|  |  |
|---|---|
| Author | J R Durgaprasad |
| Email | jr.durgaprasad@gmail.com |
| Github Link | https://github.com/jrdurgaprasad/mp2.git |
| Branch to pull | main |
|  |  |

## Project Context

A standalone end-to-end RAG build that mirrors the capstone's core loop at small scale, on a corpus.

## Objective

A small Python script that lets a user ask questions about a collection of Sherlock Holmes stories. The script:
1.Loads 5 short stories from corpus/
2.Chunks each story into manageable pieces
3.Embeds the chunks using OpenAI's text-embedding-3-small
4.Stores them in a Qdrant collection
5.Takes a question, retrieves the most relevant chunks, and answers using gpt-4o-mini — citing the source story and section
You'll then validate your pipeline against 2 predefined questions and extend it with 3 questions of your own. Submit the working script, your 5 Q&A pairs, and a short reflection.

## Project Structure

The mini project contains the following file hierarchy:

```text
mp2/
├── .env
├── README.md
├── requirements.txt
├── mp2_rag.py
├── mp2_reflection.md
└── data/learner_questions.jsonl
```

## Instructions to Run the Notebook

1. Open a terminal in the project folder:

   ```bash
   cd mp2/
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   # Edit .env with your OPENAI_API_KEY and QDRANT_URL
   source .env
   ```

2. Run the python file mp2_rag.py to validate using below command since vector db qdrant is up and running already.

   ```bash
   python mp2_rag.py validate
   ```

3. Validate the python command output with the questions mentioned in data/learner_questions.jsonl