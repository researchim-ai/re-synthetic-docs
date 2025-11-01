from dataclasses import dataclass


_JSON_SCHEMA_EXAMPLE = (
    '{\n'
    '  "text": "<весь распознанный текст>",\n'
    '  "bboxes": [\n'
    '    {"label": "signature", "bbox": [x1, y1, x2, y2]},\n'
    '    {"label": "stamp",     "bbox": [x1, y1, x2, y2]}\n'
    '  ],\n'
    '  "coord_space": "pixel"\n'
    '}'
)


@dataclass(frozen=True)
class TextBBoxPrompts:
    """Промпты для совместного распознавания текста и bbox-объектов."""

    TRAIN: str = (
        "Распознай текст документа и координаты подписи и печати. "
        "Верни ответ СТРОГО в формате JSON без комментариев и пояснений: "
        + _JSON_SCHEMA_EXAMPLE
    )

    EVAL: str = (
        "Распознай текст и координаты объектов (подпись: signature, печать: stamp). "
        "Ответ верни СТРОГО как валидный JSON (без лишнего текста) с полями text, bboxes[], coord_space. "
        "Используй coord_space=\"pixel\". Пример структуры: "
        + _JSON_SCHEMA_EXAMPLE
    )


