"""Unified offline QA system: the domain experts + retrieval + optional local LLM.

This closes the loop (roadmap #1). A question flows:

    question
      -> pick the DOMAIN expert (escolar-PT / java / dotnet / js) -- a retrieval-based gate
      -> route INSIDE it (ModularRouter: area -> sector -> subsector)
      -> retrieve the grounded passages (SparseKB, hybrid keyword+semantic)
      -> ANSWER: fluent via a local LLM (Ollama) if it is up, else EXTRACTIVE
         (the most relevant lesson, verbatim).

So the system always answers **offline** with grounded, correct content, and simply
**upgrades to fluent phrasing** when a local llama/mistral is available -- no retrain,
no network required for the base case.

Each expert lives in its own directory (as written by the `build_*_experts.py` scripts):
`<dir>/router.gate.json` (+ per-area files) and `<dir>/kb.sparse.json.gz`.
"""

import os

from . import generate as _gen


class Expert:
    """One domain: a ModularRouter (routing) + a SparseKB (passages)."""

    def __init__(self, name, router, kb):
        self.name = name
        self.router = router
        self.kb = kb

    @classmethod
    def load(cls, directory, name=None, *, router_prefix="router",
             kb_name="kb.sparse.json.gz"):
        from . import modular, vindex
        name = name or os.path.basename(os.path.normpath(directory))
        router = modular.ModularRouter().load(os.path.join(directory, router_prefix))
        kb = vindex.SparseKB().load(os.path.join(directory, kb_name))
        return cls(name, router, kb)

    def retrieve(self, question, top_k=3, multi=2):
        return self.kb.answer(question, router=self.router, top_k=top_k, multi=multi)

    def top_score(self, question):
        """Best single-passage relevance -- used as this domain's confidence."""
        _, passages = self.kb.answer(question, router=self.router, top_k=1, multi=0)
        return passages[0][2] if passages else 0.0

    def __repr__(self):
        return f"Expert({self.name!r}, passages={len(self.kb.texts)})"


class AxonSystem:
    """Routes a question across several domain experts, then answers it (RAG).

    The domain gate is a small Naive-Bayes classifier over every expert's passages
    (label = expert name). Programming languages/domains have very distinct vocabulary
    (`def/self/import` vs `public class` vs `using` vs `const/function`), so a WordNB
    over the lessons separates them far better than comparing raw retrieval scores --
    which confuse generic terms ("rota", "JSON", "RandomForest") across domains."""

    def __init__(self, experts=None):
        self.experts = list(experts or [])
        self.domain_gate = None      # WordNB: text -> expert name

    def add(self, expert):
        self.experts.append(expert)
        return self

    def fit_domain_gate(self):
        """Train the NB domain gate from the loaded experts' passages. Cheap (counting);
        no expert is retrained. Call after loading/adding experts."""
        from ._textclf import WordNB
        gate = WordNB(alpha=1.0)
        for e in self.experts:
            for text in e.kb.texts:
                gate.partial_fit(text, e.name)
        self.domain_gate = gate
        return self

    @classmethod
    def load(cls, root, names=None):
        """Load every expert under `root` (a dir of expert dirs). Pass `names` to pick
        a subset/order; otherwise auto-discovers dirs that look like an expert."""
        names = names or cls._discover(root)
        experts = [Expert.load(os.path.join(root, n), n) for n in names]
        return cls(experts).fit_domain_gate()

    @staticmethod
    def _discover(root):
        found = []
        for n in sorted(os.listdir(root)):
            d = os.path.join(root, n)
            if os.path.isfile(os.path.join(d, "router.gate.json")) and \
               os.path.isfile(os.path.join(d, "kb.sparse.json.gz")):
                found.append(n)
        return found

    def route(self, question, *, alpha=0.5):
        """Pick the domain expert. Combines TWO signals, summed (`alpha` weights them):
          - retrieval: the expert's best passage cosine (good at specifics), normalized;
          - domain gate: WordNB P(expert | text) (good at language/vocabulary).
        `alpha`=0 -> retrieval only (old behavior), 1 -> NB only, 0.5 -> the blend.
        Returns (winner_expert, [(expert, combined_score), ...] sorted desc)."""
        if not self.experts:
            return None, []
        ret = {e: e.top_score(question) for e in self.experts}
        rmax = max(ret.values()) or 1.0
        ret = {e: s / rmax for e, s in ret.items()}            # normalize to [0,1]
        proba = self.domain_gate.predict_proba(question) if self.domain_gate else {}
        combined = {e: alpha * proba.get(e.name, 0.0) + (1 - alpha) * ret[e]
                    for e in self.experts}
        scored = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return scored[0][0], scored

    def answer(self, question, *, model=_gen.DEFAULT_MODEL, top_k=3, multi=2,
               min_score=0.05, offline_ok=True):
        """Full pipeline: pick the domain, route, retrieve, answer (fluent or extractive).

        Returns a dict: expert, route, passages (path+score), answer, mode, scores.
        `mode` is "generated" (LLM), "extractive" (offline fallback) or "abstain"
        (nothing relevant / below `min_score`)."""
        if not self.experts:
            return {"expert": None, "route": None, "passages": [], "answer": None,
                    "mode": "empty", "scores": {}}
        expert, scored = self.route(question)
        scores = {e.name: round(s, 4) for e, s in scored}
        paths, passages = expert.retrieve(question, top_k=top_k, multi=multi)
        best = passages[0][2] if passages else 0.0
        if not passages or best < min_score:
            return {"expert": expert.name, "route": None, "passages": [],
                    "answer": "Não tenho informação suficiente sobre isso.",
                    "mode": "abstain", "scores": scores}
        text, mode = _gen.grounded_answer(question, passages, model=model,
                                          offline_ok=offline_ok)
        used = [{"path": p, "score": round(s, 4)} for _, p, s in passages]
        return {"expert": expert.name, "route": paths, "passages": used,
                "answer": text, "mode": mode, "scores": scores}
