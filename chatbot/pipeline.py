import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_MODEL = None
_TOKENIZER = None


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    model_id = os.getenv("MODEL_ID", "exaone-4.0-1.2b")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _MODEL.eval()
    return _MODEL, _TOKENIZER


def _build_prompt(message, history, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if history:
            for item in history:
                role = item.get("role", "user")
                content = item.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = []
    if history:
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            if content:
                prompt.append(f"{role.capitalize()}: {content}")
    prompt.append(f"User: {message}")
    prompt.append("Assistant:")
    return "\n".join(prompt)


def get_pipeline():
    def _pipeline(message, history=None):
        model, tokenizer = _load()
        prompt = _build_prompt(message, history, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        top_p = float(os.getenv("TOP_P", "0.9"))
        do_sample = temperature > 0

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()

    return _pipeline
