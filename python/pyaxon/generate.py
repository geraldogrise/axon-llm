"""Generation layer: RAG-grounded fluent answers via a local LLM (Ollama).

Closes the loop: axon (router + KB) finds the right facts, a local open model writes
the fluent answer grounded in them -- so it is both fluent AND correct. Runs fully
offline once the model is pulled (`ollama pull <model>`).

Flow: route (with a confidence gate) -> retrieve passages from that compartment ->
build a grounded prompt -> call Ollama -> return the answer. If confidence is too low,
it abstains ("não sei") instead of hallucinating.
"""

import json
import urllib.error
import urllib.request

DEFAULT_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

_PROMPT = """Você é um assistente que responde em português do Brasil.
Responda à PERGUNTA usando SOMENTE o CONTEXTO abaixo. Se o contexto não bastar, diga
que não tem informação suficiente. Seja claro e didático.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def ollama_generate(prompt, model=DEFAULT_MODEL, url=OLLAMA_URL, timeout=120,
                    temperature=0.3):
    """Call a local Ollama server. Returns the generated text (raises on failure)."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature}}).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "").strip()


def is_ollama_up(url=OLLAMA_TAGS, timeout=1.5):
    """Quick probe: is a local Ollama server reachable? Never raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def build_context(passages, max_context_chars=1800):
    """Concatenate retrieved passages into a grounding context (+ the ones used)."""
    context, used = "", []
    for chunk, cpath, score in passages:
        if len(context) + len(chunk) > max_context_chars:
            break
        context += chunk + "\n\n"
        used.append({"path": cpath, "score": score})
    return context.strip(), used


def extractive_answer(passages, max_chars=700):
    """Offline fallback: return the most relevant retrieved lesson verbatim (trimmed).
    No LLM needed -- grounded and correct, just not rephrased."""
    if not passages:
        return None
    text = passages[0][0].strip().replace("\n\n", "\n")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def grounded_answer(question, passages, *, model=DEFAULT_MODEL, offline_ok=True,
                    max_context_chars=1800):
    """Answer grounded in `passages`. Fluent via Ollama if it is up; otherwise falls
    back to an EXTRACTIVE answer (the top lesson) so it still answers offline.
    Returns (text, mode) where mode is "generated" | "extractive"."""
    context, _ = build_context(passages, max_context_chars)
    if model and is_ollama_up():
        try:
            prompt = _PROMPT.format(context=context, question=question)
            return ollama_generate(prompt, model=model), "generated"
        except (urllib.error.URLError, OSError, ValueError):
            if not offline_ok:
                raise
    if not offline_ok:
        raise RuntimeError("Ollama unavailable and offline_ok=False")
    return extractive_answer(passages), "extractive"


def answer(question, kb, router, *, model=DEFAULT_MODEL, top_k=3, multi=2,
           threshold=0.0, offline_ok=True, max_context_chars=1800):
    """RAG-grounded answer over one (kb, router). Returns a dict with the route, the
    passages used, the answer text and the `mode` ("generated"/"extractive"/"abstain").
    Abstains when routing confidence < threshold; degrades to extractive when Ollama
    is down (so it answers even without a local LLM)."""
    # confidence gate (robustness): don't answer if unsure
    conf = router.confidence(question) if hasattr(router, "confidence") else 1.0
    if threshold > 0 and conf < threshold:
        return {"route": None, "confidence": conf, "passages": [],
                "answer": "Não tenho informação suficiente sobre isso.",
                "abstained": True, "mode": "abstain"}

    paths, passages = kb.answer(question, router=router, top_k=top_k, multi=multi)
    _, used = build_context(passages, max_context_chars)
    text, mode = grounded_answer(question, passages, model=model, offline_ok=offline_ok,
                                 max_context_chars=max_context_chars)
    return {"route": paths, "confidence": conf, "passages": used,
            "answer": text, "abstained": False, "mode": mode}
