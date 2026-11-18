# Intern Task Assistant

A small Generative AI application built for **Day 16** of the Big Brains Generative AI internship.
It answers questions about internship tasks, submission rules and GenAI concepts **only from its own knowledge base**, and says so clearly when the answer is not there.

## Problem and users
- **Problem:** task steps, rules and concepts are spread across long pages, and a general chatbot may guess.
- **Target users:** interns who need quick, source-backed answers.
- **Main feature:** a chat that returns an answer, its sources and a confidence level.

## GenAI concepts used
| Concept | Where |
|---|---|
| LLM API + prompt engineering | `SYSTEM_PROMPT` in `rag_assistant.py` (role, rules, output format) |
| RAG | Embeddings + cosine similarity over `knowledge_base/*.txt` |
| Structured JSON output | `answer`, `found_in_knowledge_base`, `sources`, `confidence` |
| Function calling | `days_until` tool (exact date maths in Python) |
| Evaluation / guardrails | `validate()` checks schema, fake sources and confidence |

## Workflow
Question -> embed and retrieve top 3 chunks -> LLM answers from context (may call `days_until`) -> parse and validate JSON -> show answer, sources and confidence.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
```
Open `.env` and paste your API key into `LLM_API_KEY`. A free Gemini key from Google AI Studio works with the default settings.

## Run
```bash
streamlit run app.py
```

## Test
```bash
python tests.py          # offline unit tests, no API key needed
python tests.py --live   # 7 live test cases, saves test_report.json
```
The 7 live cases cover normal questions, a tool call, unavailable information, an off-topic request, Roman Urdu and an invalid date.

## Project structure
```
app.py              Streamlit chat UI
rag_assistant.py    RAG, prompt, tool calling, validation
tests.py            offline + live tests
knowledge_base/     .txt files the assistant answers from
.env.example        copy to .env and add your key
```

## Security
Never commit `.env` or show your API key in screenshots. `.gitignore` already excludes it.

## Improve it
Add more `.txt` files to `knowledge_base/`. The app re-reads them on every start.
