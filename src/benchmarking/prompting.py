from __future__ import annotations

import hashlib
import os
from typing import Dict


DEFAULT_TEMPLATES = {
    "Zero-Shot": "{input_section}\nFRAGE: {question}",
    "Advanced-Prompt": "Du bist ein erfahrener Senior ESG Consultant.\n{input_section}\nFRAGE: {question}",
    "RAG": (
        "Nutze ZUSAETZLICHES WISSEN aus der Datenbank, um die Frage zu beantworten:\n\n"
        "KONTEXT:\n{rag_context}\n\n{input_section}\nFRAGE: {question}"
    ),
}


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_prompt_templates(paths: Dict[str, str]) -> Dict[str, str]:
    templates = dict(DEFAULT_TEMPLATES)
    for method, path in paths.items():
        if path and os.path.exists(path):
            content = _read_file(path)
            if content:
                templates[method] = content
    return templates


def prompt_templates_hash(templates: Dict[str, str]) -> str:
    payload = "\n".join([f"{k}\n{templates.get(k,'')}" for k in sorted(templates.keys())])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt(
    method: str,
    question: str,
    input_doc_text: str = "",
    rag_context: str = "",
    answer_type: str = "",
    max_words: str = "",
    atomic_facts=None,
    prompt_templates: Dict[str, str] | None = None,
) -> str:
    input_section = ""
    if input_doc_text:
        input_section = f"\nACHTUNG - DOKUMENT/INPUT:\n{input_doc_text}\n"

    templates = prompt_templates or DEFAULT_TEMPLATES
    template = templates.get(method, "{input_section}\nFRAGE: {question}")
    try:
        prompt = template.format(
            question=question,
            input_section=input_section,
            rag_context=rag_context,
        )
    except Exception:
        prompt = f"{input_section}\nFRAGE: {question}"

    answer_type_norm = str(answer_type).strip().lower()
    has_atomic_target = bool(atomic_facts and len(atomic_facts) > 0)
    is_binary = "wahr" in answer_type_norm or "falsch" in answer_type_norm
    is_atomic = "atomic" in answer_type_norm or "fakt" in answer_type_norm
    is_free_text = "frei" in answer_type_norm or "free" in answer_type_norm

    if is_binary:
        prompt += "\n\nAntworte ausschliesslich mit 'Wahr' oder 'Falsch'."
    elif is_atomic:
        prompt += (
            "\n\nAntworte ausschliesslich als Liste von Atomic Facts "
            "(ein Fakt pro Zeile; praezise, pruefbar, ohne Einleitung)."
        )
    elif is_free_text:
        prompt += "\n\nAntworte als kurzen, zusammenhaengenden Freitext."

    if has_atomic_target and not is_binary:
        prompt += (
            "\n\nFormatiere die Antwort als Liste von Atomic Facts "
            "(pro Zeile genau ein knapper, pruefbarer Fakt)."
        )

    max_words_text = str(max_words).strip()
    if max_words_text:
        if max_words_text.isdigit():
            prompt += f"\n\nFasse dich kurz. Max {max_words_text} Woerter."
        else:
            prompt += f"\n\nHalte die Antwort kurz. Zusatzregel zur Laenge: {max_words_text}"
    return prompt
