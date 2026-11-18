import streamlit as st

from rag_assistant import Assistant

st.set_page_config(page_title="Intern Task Assistant", page_icon="🤖")
st.title("🤖 Intern Task Assistant")
st.caption("RAG + structured output + function calling. Ask about the Day 16 task or GenAI concepts.")


@st.cache_resource
def get_assistant():
    return Assistant()


try:
    assistant = get_assistant()
except Exception as e:
    st.error(str(e))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Thinking..."):
        out = assistant.ask(question, st.session_state.history)
    res = out["result"]

    with st.chat_message("assistant"):
        st.write(res["answer"])
        st.caption(
            f"Found in KB: {res['found_in_knowledge_base']} | Confidence: {res['confidence']} | "
            f"Sources: {', '.join(res['sources']) or 'none'}"
        )
        for w in out["warnings"]:
            st.warning(w)
        if out["tool_calls"]:
            st.info(f"Tool used: {out['tool_calls']}")
        with st.expander("Retrieved context"):
            for c in out["retrieved"]:
                st.markdown(f"**{c['source']}** (score {c['score']:.2f})")
                st.write(c["text"])
        with st.expander("Raw JSON"):
            st.json(res)

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": res["answer"]})

with st.sidebar:
    st.header("About")
    st.write("Answers only from the knowledge base. If the info is missing, it says so.")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()
