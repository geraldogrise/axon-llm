"""Language identification -- the first gate of axon-lang.

Before routing by area/sector, axon-lang needs to know it is talking in
**Portuguese**. This module trains (builds char n-gram profiles) from samples of
several languages and classifies new texts.
"""

import json

from ._textclf import ProfileClassifier

# Short samples per language (enough for trigram profiles). Can be extended with
# the pyaxon.corpus pipeline (more text = better profiles).
_SAMPLES = {
    "pt": (
        "o rato roeu a roupa do rei de roma. a matemática é a ciência dos números e das "
        "formas. hoje o dia está ensolarado e vamos estudar teoria dos números. a solução "
        "do problema depende de álgebra e de cálculo. quando chove a rua fica molhada. "
        "precisamos entender a questão antes de resolver o exercício de física."
    ),
    "en": (
        "the quick brown fox jumps over the lazy dog. mathematics is the science of numbers "
        "and shapes. today the weather is sunny and we will study number theory. the solution "
        "to the problem depends on algebra and calculus. when it rains the street gets wet. "
        "we need to understand the question before solving the physics exercise."
    ),
    "es": (
        "el zorro marrón salta sobre el perro perezoso. las matemáticas son la ciencia de los "
        "números y las formas. hoy hace sol y vamos a estudiar teoría de números. la solución "
        "del problema depende del álgebra y del cálculo. cuando llueve la calle se moja. "
        "necesitamos entender la pregunta antes de resolver el ejercicio de física."
    ),
    "fr": (
        "le renard brun saute par-dessus le chien paresseux. les mathématiques sont la science "
        "des nombres et des formes. aujourd'hui il fait soleil et nous allons étudier la théorie "
        "des nombres. la solution du problème dépend de l'algèbre et du calcul. quand il pleut "
        "la rue est mouillée. nous devons comprendre la question avant de résoudre l'exercice."
    ),
    "it": (
        "la volpe marrone salta sopra il cane pigro. la matematica è la scienza dei numeri e "
        "delle forme. oggi c'è il sole e studieremo la teoria dei numeri. la soluzione del "
        "problema dipende dall'algebra e dal calcolo. quando piove la strada si bagna. "
        "dobbiamo capire la domanda prima di risolvere l'esercizio di fisica."
    ),
}


class LanguageIdentifier:
    def __init__(self, n=3, top=400):
        self._clf = ProfileClassifier(n=n, top=top)

    def fit(self, samples=None):
        """Train with {language: text}. Without an argument, uses the built-in samples."""
        self._clf.fit(samples if samples is not None else _SAMPLES)
        return self

    def predict(self, text):
        return self._clf.predict(text)

    def predict_proba(self, text):
        return self._clf.predict_proba(text)

    def is_portuguese(self, text, threshold=0.0):
        """True if the **most likely** language is Portuguese (argmax).

        threshold>0 also requires a minimum confidence. The default (0.0) uses only
        the argmax, robust for short sentences where pt/es/it split the probability."""
        return self.predict(text) == "pt" and self._clf.predict_proba(text).get("pt", 0.0) >= threshold

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"n": self._clf.n, "top": self._clf.top,
                       "profiles": self._clf.profiles_}, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self._clf.n = d["n"]
        self._clf.top = d["top"]
        self._clf.profiles_ = d["profiles"]
        return self


def detect(text):
    """Shortcut: identify the language of a text (trains with built-in samples)."""
    return LanguageIdentifier().fit().predict(text)
