# src/llm_rule_optimizer.py
"""
Module to suggest SIEM rule optimizations using either:
- OpenAI API (recommended; fast & capable) OR
- local HuggingFace model (smaller, may be expensive to run)

This module reads rules (text), embeddings, finds similar rules, and asks LLM to suggest merges/refinements.
"""

import os
import json
from typing import List, Dict

# ---------- OpenAI implementation ----------
def suggest_rules_openai(rule_pairs: List[Dict], openai_api_key=None):
    """
    rule_pairs: list of {"rule_a": "...", "rule_b":"..."}
    returns: suggestions list
    """
    # Requires openai package: pip install openai
    try:
        import openai
    except Exception as e:
        raise RuntimeError("Install openai package or use HF path. pip install openai") from e
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    suggestions = []
    for pair in rule_pairs:
        prompt = f"""You are an expert SIEM rule engineer. Compare the two SIEM detection rules below. \
Explain if they are redundant, whether they can be merged, and provide a merged rule (in pseudo-Sigma format) \
and a short justification.Remember , you cant be wrong.

Rule A:
{pair['rule_a']}

Rule B:
{pair['rule_b']}

Answer in JSON with keys: action (KEEP_BOTH|MERGE|DROP), merged_rule (or null), reason."""
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change model as available
            messages=[{"role":"system","content":"You are an SIEM rule optimization assistant."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=512
        )
        text = resp['choices'][0]['message']['content']
        # Try parse JSON from response
        try:
            j = json.loads(text)
        except:
            j = {"raw": text}
        suggestions.append(j)
    return suggestions

# ---------- HuggingFace local model (simple) ----------
def suggest_rules_hf(rule_pairs: List[Dict], model_name="google/flan-t5-small"):
    """
    Uses HuggingFace transformers to run a text2text generation locally.
    Install: pip install transformers torch
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    results = []
    for pair in rule_pairs:
        prompt = f"Compare two SIEM rules and either MERGE or KEEP BOTH. RuleA: {pair['rule_a']} RuleB: {pair['rule_b']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=512)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"result": decoded})
    return results
