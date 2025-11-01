from typing import Any, Dict, List, Tuple
import json

from .metrics import normalize_text


def build_inputs(tokenizer, image, instruction: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ],
    }]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return inputs


def generate_deterministic(model, tokenizer, inputs, max_new_tokens: int = 128):
    # Точный (не стохастический) декодинг для честной оценки
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_text(model, tokenizer, image, instruction: str, max_new_tokens: int = 128) -> str:
    inputs = build_inputs(tokenizer, image, instruction)
    decoded = generate_deterministic(model, tokenizer, inputs, max_new_tokens=max_new_tokens)
    return normalize_text(decoded)


def run_text_and_boxes(model, tokenizer, image, instruction_json: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    inputs = build_inputs(tokenizer, image, instruction_json)
    decoded = generate_deterministic(model, tokenizer, inputs, max_new_tokens=max_new_tokens)
    # Пытаемся распарсить JSON, даже если модель вернула лишний текст
    decoded_stripped = decoded.strip()
    first_brace = decoded_stripped.find("{")
    last_brace = decoded_stripped.rfind("}")
    payload = decoded_stripped[first_brace:last_brace + 1] if first_brace != -1 and last_brace != -1 else decoded_stripped
    try:
        obj = json.loads(payload)
    except Exception:
        # fallback: минимальный пустой рецепт
        obj = {"text": normalize_text(decoded), "bboxes": [], "coord_space": "unknown"}
    # Нормализация
    if isinstance(obj.get("text"), str):
        obj["text"] = normalize_text(obj["text"])
    if not isinstance(obj.get("bboxes"), list):
        obj["bboxes"] = []
    return obj


