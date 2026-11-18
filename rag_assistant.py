"""
Intern Task Assistant - core logic.

GenAI concepts used:
1. LLM API + prompt engineering (role, rules, output format)
2. RAG (embeddings + cosine-similarity retrieval over a small knowledge base)
3. Structured JSON output with validation
4. Function calling (one tool: days_until)
5. Simple evaluation / guardrail check on every answer
"""
import glob
import json
import os
import re
from datetime import date, datetime

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
KB_DIR = os.getenv("KB_DIR", "knowledge_base")

SYSTEM_PROMPT = """You are "Intern Task Assistant", a helpful assistant for interns in a Generative AI internship.

ROLE
- Answer questions about internship tasks, submission rules and Generative AI concepts.

RULES
1. Answer ONLY from the CONTEXT given in the user message. Never invent facts, dates, numbers or policies.
2. If the CONTEXT does not contain the answer, set "found_in_knowledge_base" to false and say clearly that the information is not available in the knowledge base.
3. If the question is unrelated to the internship or Generative AI (e.g. poems, sports), politely refuse and set "found_in_knowledge_base" to false.
4. If the user asks how many days remain until a date, call the days_until tool. Do not calculate dates yourself.
5. Reply in the same language style as the user (English, Urdu or Roman Urdu).
6. Be concise: maximum 5 sentences.

OUTPUT FORMAT
Return ONLY a valid JSON object, with no markdown fences and no extra text:
{
  "answer": "string",
  "found_in_knowledge_base": true or false,
  "sources": ["file names from the CONTEXT that you used"],
  "confidence": "high" | "medium" | "low"
}
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "days_until",
            "description": "Returns how many days are left from today until a target date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_date": {
                        "type": "string",
                        "description": "Target date in YYYY-MM-DD format.",
                    }
                },
                "required": ["target_date"],
            },
        },
    }
]


# ----------------------------- Tool -----------------------------
def days_until(target_date: str) -> dict:
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid date. Use YYYY-MM-DD."}
    today = date.today()
    return {"today": today.isoformat(), "target_date": target.isoformat(), "days_left": (target - today).days}


TOOL_REGISTRY = {"days_until": days_until}


# ----------------------------- RAG -----------------------------
def load_chunks(kb_dir: str = KB_DIR):
    """Split every .txt file into paragraph chunks."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for part in re.split(r"\n\s*\n", text):
            part = part.strip()
            if len(part) > 30:
                chunks.append({"source": name, "text": part})
    return chunks


def pick_model(available, wanted, kind):
    """Return `wanted` if the account has it, otherwise the best matching model the account does have.

    kind is "chat" or "embed". This avoids 404 errors when a model name is retired or mistyped.
    """
    names = [m.split("/", 1)[-1] for m in available]
    if wanted in names:
        return wanted
    if kind == "embed":
        cands = [n for n in names if "embedding" in n]
        if "gemini-embedding-001" in cands:
            return "gemini-embedding-001"
        return cands[0] if cands else wanted
    bad = ("embedding", "tts", "image", "live", "audio", "vision", "robotics", "computer", "learnlm", "gemma")
    cands = [n for n in names if n.startswith("gemini") and not any(b in n for b in bad)]
    stable = [n for n in cands if not any(b in n for b in ("preview", "exp", "lite", "thinking"))]
    for group in (
        [n for n in stable if "flash" in n],
        [n for n in cands if "flash" in n and "lite" in n and "preview" not in n],
        stable,
        cands,
    ):
        if group:
            return sorted(group, reverse=True)[0]
    return wanted


class Assistant:
    def __init__(self):
        if not API_KEY:
            raise RuntimeError("LLM_API_KEY is missing. Copy .env.example to .env and add your key.")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=6)
        self.chat_model, self.embed_model = CHAT_MODEL, EMBED_MODEL
        try:  # use a model the account really has, instead of trusting the .env name
            available = [m.id for m in self.client.models.list()]
            self.chat_model = pick_model(available, CHAT_MODEL, "chat")
            self.embed_model = pick_model(available, EMBED_MODEL, "embed")
        except Exception as e:
            print(f"[warning] could not list models, using .env names: {e}")
        print(f"[models] chat={self.chat_model} embed={self.embed_model}")
        self.chunks = load_chunks()
        if not self.chunks:
            raise RuntimeError(f"No .txt files found in '{KB_DIR}/'.")
        self.vectors = self._embed([c["text"] for c in self.chunks])

    def _embed(self, texts):
        vectors = []
        for i in range(0, len(texts), 20):  # batch to stay under API limits
            resp = self.client.embeddings.create(model=self.embed_model, input=texts[i:i + 20])
            vectors.extend(d.embedding for d in resp.data)
        arr = np.array(vectors, dtype="float32")
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def retrieve(self, question: str):
        q = self._embed([question])[0]
        scores = self.vectors @ q
        order = np.argsort(scores)[::-1][:TOP_K]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order if scores[i] >= MIN_SCORE]

    # ------------------------- Chat -------------------------
    def ask(self, question: str, history=None):
        history = history or []
        retrieved = self.retrieve(question)
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in retrieved) or "(no relevant context found)"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history[-6:]:  # short conversation memory
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})

        tool_calls_made = []
        content = ""
        for _ in range(3):  # tool-calling loop
            resp = self.client.chat.completions.create(
                model=self.chat_model, messages=messages, tools=TOOLS, temperature=0.2
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append(msg)
                for call in msg.tool_calls:
                    fn = TOOL_REGISTRY.get(call.function.name)
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        result = fn(**args) if fn else {"error": "Unknown tool"}
                    except Exception as e:  # bad arguments from the model
                        result = {"error": str(e)}
                    tool_calls_made.append({"tool": call.function.name, "result": result})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                continue
            content = msg.content or ""
            break

        data = self._parse_json(content)
        if data is None:  # retry once with a stricter instruction
            messages.append({"role": "user", "content": "Return ONLY the JSON object, nothing else."})
            resp = self.client.chat.completions.create(model=self.chat_model, messages=messages, temperature=0)
            data = self._parse_json(resp.choices[0].message.content or "")
        if data is None:
            data = {"answer": "Sorry, I could not produce a valid answer.", "found_in_knowledge_base": False,
                    "sources": [], "confidence": "low"}

        data, warnings = self.validate(data, retrieved, tool_calls_made)
        return {"result": data, "retrieved": retrieved, "tool_calls": tool_calls_made, "warnings": warnings}

    # ------------------- Parsing + evaluation -------------------
    @staticmethod
    def _parse_json(text: str):
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def validate(data: dict, retrieved: list, tool_calls: list):
        """Guardrail checks: schema, hallucinated sources, unsupported 'found' claims."""
        warnings = []
        data.setdefault("answer", "")
        data["found_in_knowledge_base"] = bool(data.get("found_in_knowledge_base", False))
        data["sources"] = data.get("sources") if isinstance(data.get("sources"), list) else []
        if data.get("confidence") not in ("high", "medium", "low"):
            data["confidence"] = "low"
            warnings.append("Invalid confidence value, reset to 'low'.")

        valid_sources = {c["source"] for c in retrieved}
        fake = [s for s in data["sources"] if s not in valid_sources]
        if fake:
            data["sources"] = [s for s in data["sources"] if s in valid_sources]
            warnings.append(f"Removed sources not in retrieved context: {fake}")

        if data["found_in_knowledge_base"] and not data["sources"] and not tool_calls:
            data["confidence"] = "low"
            warnings.append("Answer claims to be found but cites no source; confidence lowered.")
        if not data["found_in_knowledge_base"]:
            data["sources"] = []
        return data, warnings