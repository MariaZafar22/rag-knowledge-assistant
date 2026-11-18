"""
Test runner for the Intern Task Assistant.

  python tests.py            -> offline unit tests (no API key needed)
  python tests.py --live     -> 7 live test cases against the real model (needs .env)

The live run saves test_report.json, which you can use for your screenshots and presentation.
"""
import json
import sys
from datetime import date, timedelta

import rag_assistant as ra

# ---------------------------------------------------------------- offline tests
def run_offline():
    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}")

    print("Offline unit tests")

    # tool
    future = (date.today() + timedelta(days=10)).isoformat()
    check("days_until returns correct number", ra.days_until(future)["days_left"] == 10)
    check("days_until rejects bad date", "error" in ra.days_until("18/11/2026"))

    # JSON parsing
    P = ra.Assistant._parse_json
    check("parse plain JSON", P('{"answer": "hi"}') == {"answer": "hi"})
    check("parse fenced JSON", P('```json\n{"answer": "hi"}\n```') == {"answer": "hi"})
    check("parse JSON with extra text", P('Sure! {"answer": "hi"} Done.') == {"answer": "hi"})
    check("parse garbage returns None", P("no json here") is None)

    # validation / guardrails
    V = ra.Assistant.validate
    retrieved = [{"source": "genai_concepts.txt", "text": "x", "score": 0.9}]

    d, w = V({"answer": "a", "found_in_knowledge_base": True, "sources": ["genai_concepts.txt"],
              "confidence": "high"}, retrieved, [])
    check("valid answer passes without warnings", not w and d["confidence"] == "high")

    d, w = V({"answer": "a", "found_in_knowledge_base": True, "sources": ["made_up.txt"],
              "confidence": "high"}, retrieved, [])
    check("hallucinated source is removed", d["sources"] == [] and any("Removed" in x for x in w))
    check("claim without source lowers confidence", d["confidence"] == "low")

    d, w = V({"answer": "a", "found_in_knowledge_base": False, "sources": ["genai_concepts.txt"],
              "confidence": "banana"}, retrieved, [])
    check("invalid confidence reset to low", d["confidence"] == "low")
    check("not-found answer has no sources", d["sources"] == [])

    d, w = V({"answer": "5 days", "found_in_knowledge_base": True, "sources": [], "confidence": "high"},
             [], [{"tool": "days_until", "result": {}}])
    check("tool-based answer keeps confidence", d["confidence"] == "high")

    # knowledge base loading
    chunks = ra.load_chunks()
    check("knowledge base loads chunks", len(chunks) >= 10)

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------- live tests
LIVE_CASES = [
    {
        "id": 1, "type": "normal",
        "question": "How many inputs do I need to test my application with?",
        "expect_found": True, "must_contain": ["five", "5"],
    },
    {
        "id": 2, "type": "normal",
        "question": "What is RAG and why is it useful?",
        "expect_found": True, "must_contain": ["retriev"],
    },
    {
        "id": 3, "type": "tool (function calling)",
        "question": f"How many days are left until {(date.today() + timedelta(days=30)).isoformat()}?",
        "expect_tool": "days_until",
    },
    {
        "id": 4, "type": "edge: information unavailable",
        "question": "What is the stipend amount for the Big Brains internship?",
        "expect_found": False,
    },
    {
        "id": 5, "type": "edge: off-topic",
        "question": "Write me a poem about cricket.",
        "expect_found": False,
    },
    {
        "id": 6, "type": "edge: Roman Urdu",
        "question": "LinkedIn post mein mujhe kya kya likhna hai?",
        "expect_found": True, "must_contain": [],
    },
    {
        "id": 7, "type": "edge: invalid date for tool",
        "question": "How many days until 2026-13-45?",
        "expect_tool": "days_until",
    },
]


def run_live():
    assistant = ra.Assistant()
    report, passed = [], 0
    for case in LIVE_CASES:
        out = assistant.ask(case["question"])
        res = out["result"]
        ok, reasons = True, []

        if "expect_found" in case and res["found_in_knowledge_base"] != case["expect_found"]:
            ok = False
            reasons.append(f"found_in_knowledge_base should be {case['expect_found']}")
        if "expect_tool" in case and not any(t["tool"] == case["expect_tool"] for t in out["tool_calls"]):
            ok = False
            reasons.append(f"tool {case['expect_tool']} was not called")
        words = case.get("must_contain", [])
        if words and not any(w.lower() in res["answer"].lower() for w in words):
            ok = False
            reasons.append(f"answer should mention one of {words}")

        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{case['id']} ({case['type']}): {case['question']}")
        print(f"        answer: {res['answer']}")
        print(f"        found={res['found_in_knowledge_base']} confidence={res['confidence']} "
              f"sources={res['sources']} tools={[t['tool'] for t in out['tool_calls']]}")
        for r in reasons:
            print(f"        problem: {r}")
        report.append({"case": case, "passed": ok, "reasons": reasons, "result": res,
                       "tool_calls": out["tool_calls"], "warnings": out["warnings"]})

    print(f"\n{passed}/{len(LIVE_CASES)} live cases passed")
    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Saved test_report.json")
    return passed == len(LIVE_CASES)


if __name__ == "__main__":
    ok = run_live() if "--live" in sys.argv else run_offline()
    sys.exit(0 if ok else 1)
