# **RAG Knowledge Assistant**

**A grounded Q&A assistant that answers only from its own knowledge base, and says so clearly when the answer is not there.**

Built with **Python, Streamlit and Google Gemini** (via the OpenAI-compatible endpoint).

---

## **Problem and Users**

- **Problem:** Information is spread across long documents, and a general chatbot may guess or hallucinate.
- **Target users:** Anyone who needs quick, source-backed answers from a fixed set of documents.
- **Main feature:** A chat that returns an **answer**, its **sources** and a **confidence level**.

---

## **GenAI Concepts Used**

| **Concept** | **Where** |
|---|---|
| **LLM API + prompt engineering** | `SYSTEM_PROMPT` in `rag_assistant.py` (role, rules, output format) |
| **RAG** | Embeddings + cosine similarity over `knowledge_base/*.txt` |
| **Structured JSON output** | `answer`, `found_in_knowledge_base`, `sources`, `confidence` |
| **Function calling** | `days_until` tool (exact date maths in Python) |
| **Evaluation / guardrails** | `validate()` checks schema, fake sources and confidence |

---

## **Workflow**

**Question** → embed and retrieve top 3 chunks → LLM answers from context (may call `days_until`) → parse and validate JSON → show **answer, sources and confidence**.

---

## **Setup**

```bash
python -m venv venv
venv\Scripts\activate          # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
```

Open `.env` and paste your API key into `LLM_API_KEY`. A Gemini key from Google AI Studio works with the default settings.

---

## **Run**

```bash
streamlit run app.py
```

---

## **Test**

```bash
python tests.py          # offline unit tests, no API key needed
python tests.py --live   # 7 live test cases, saves test_report.json
```

The 7 live cases cover **normal questions, a tool call, unavailable information, an off-topic request, Roman Urdu and an invalid date**.

---

## **Project Structure**

```
app.py              Streamlit chat UI
rag_assistant.py    RAG, prompt, tool calling, validation
tests.py            offline + live tests
knowledge_base/     .txt files the assistant answers from
.env.example        copy to .env and add your key
```

---

## **Security**

**Never commit `.env` or show your API key in screenshots.** `.gitignore` already excludes it.

---

## **Improve It**

Add more `.txt` files to `knowledge_base/`. The app re-reads them on every start.

---

## **Author**

**Maria Zafar**
Computer Science student | AI/ML and app developer

**GitHub:** [MariaZafar22](https://github.com/MariaZafar22)
**LinkedIn:** [MariaZafar22](https://www.linkedin.com/in/MariaZafar22)

> **An AI that admits what it does not know is more useful than one that guesses with confidence.**

If this project helped you, a ⭐ on the repo means a lot.
