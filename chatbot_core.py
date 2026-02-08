import os
from langchain_openai import ChatOpenAI

_db = None

def set_db(db):
    global _db
    _db = db

def _retrieve_docs(query: str, k: int = 4):
    if not _db:
        return []
    retriever = _db.as_retriever(search_kwargs={"k": k})

    # ✅ Support both LangChain APIs (old + new)
    try:
        return retriever.invoke(query)  # newer LC
    except Exception:
        try:
            return retriever.get_relevant_documents(query)  # older LC
        except Exception:
            return []

def generate_reply(
    user_text: str,
    extra_context: str = "",
    retrieval_query: str = "",
    use_knowledge: bool = True,
) -> str:
    """
    extra_context: anything you want the bot to prioritise (e.g. website text)
    use_knowledge: if False, the bot will NOT use FAISS retrieved docs
    """
    context_parts = []

    if extra_context:
        context_parts.append("WEBSITE / LIVE CONTEXT (highest priority):\n" + extra_context)

    if use_knowledge and retrieval_query:
        docs = _retrieve_docs(retrieval_query, k=4)
        for d in docs:
            try:
                context_parts.append("KNOWLEDGE CONTEXT:\n" + d.page_content)
            except Exception:
                pass

    context = "\n\n---\n\n".join(context_parts)[:12000]

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    prompt = f"""You are an assistant for South London College.

Rules:
- If WEBSITE / LIVE CONTEXT is present, use it as the main source of truth.
- Only use KNOWLEDGE CONTEXT if WEBSITE / LIVE CONTEXT is missing.
- If the answer is not in the context, say you don’t have that information and suggest contacting the team.
- Do NOT recommend courses unless the user explicitly asks for course recommendations.

CONTEXT:
{context}

USER:
{user_text}

ASSISTANT:
"""

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))
