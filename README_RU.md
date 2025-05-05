Thought for a couple of seconds


````markdown
# SyntheticDocs

**Набор инструментов для генерации синтетических «отсканированных» A4-документов с подписями, печатями и артефактами сканирования для обучения VLM/LLM.**

> ❗️ **Только для исследовательских целей:** этот проект предназначен исключительно для научных исследований и экспериментов. Сгенерированные документы **не должны** использоваться в реальных или производственных сценариях и ни в коем случае не применяться в мошеннических целях.

---

## Особенности

- **Текст через LLM**: локальный инференс vLLM + модели Hugging Face  
- **Шаблоны промптов** в `prompts.json` для разных типов документов  
- **Кириллический шрифт** для реалистичного русского текста в PDF  
- **Подписи и печати** с автоматическим масштабированием и прозрачностью  
- **Эффекты «сканирования»**: перспектива, размытие, JPEG-артефакты, шум (Albumentations)  
- **Выходные файлы**: 
  - `.pdf` — оригинал документа
  - `.png` — «скан» в 300 DPI
  - `.json` — метаданные (bounding-box’ы, реальный текст LLM)

---

## Структура проекта

```text
re-synthetic-docs/
├── assets/
│   ├── signatures/
│   │   ├── fonts/           # .ttf-шрифты для генерации подписей
│   │   └── png/             # готовые PNG-подписи
│   └── stamps/
│       ├── fonts/           # .ttf-шрифты для генерации печатей
│       └── png/             # готовые PNG-печати
├── prompts.json            # JSON-шаблоны промптов по типам документов
├── make_signatures.py      # скрипт генерации подписей из шрифтов
├── make_stamps.py          # скрипт генерации круглых печатей
├── syntheticdocs.py        # основной генератор: LLM → PDF → PNG → метаданные
└── requirements.txt        # зависимости Python
````

---

## Установка

```bash
git clone https://github.com/your-org/re-synthetic-docs.git
cd re-synthetic-docs

# (опционально) создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Примечание:**
>
> * Требуется Python 3.10+
> * Для GPU-ускорения предварительно установите корректную сборку `torch` + CUDA
> * На Linux может понадобиться `poppler-utils` для `pdf2image`:
>
>   ```bash
>   sudo apt update && sudo apt install -y poppler-utils
>   ```

---

## Подготовка активов

1. **Шрифты**
   Положите хотя бы один кириллический `.ttf` в
   `assets/signatures/fonts/` и `assets/stamps/fonts/`.

2. **Генерация подписей** (опционально)

   ```bash
   python make_signatures.py
   ```

   — \~500 PNG-подписи появятся в `assets/signatures/png/`.

3. **Генерация печатей** (опционально)

   ```bash
   python make_stamps.py
   ```

   — \~30 PNG-печати появятся в `assets/stamps/png/`.

---

## Использование

```bash
python syntheticdocs.py \
  --model   "/путь/к/модели/или_repo-id" \
  --font    assets/signatures/fonts/YourCyrillicFont.ttf \
  --signatures assets/signatures/png \
  --stamps  assets/stamps/png \
  --out     out \
  -n        10
```

* `--model`      — HF repo-id (например, `mistralai/Mistral-7B-Instruct-v0.2`) или локальная папка
* `--font`       — путь к TTF-шрифту с кириллицей
* `--signatures` — папка с PNG-подписями (α-канал)
* `--stamps`     — папка с PNG-печатями (α-канал)
* `--out`        — выходная директория (будет создана)
* `-n`           — количество документов

**Выход**:

```
out/
├── <uuid1>.pdf
├── <uuid1>.png
├── <uuid1>.json
├── <uuid2>.pdf
├── <uuid2>.png
├── <uuid2>.json
└── …
```

* `<uuid>.pdf`  — сгенерированный PDF
* `<uuid>.png`  — «сканированное» изображение (300 DPI)
* `<uuid>.json` — метаданные:

  ```json
  {
    "id": "<uuid>",
    "type": "<doc_type>",
    "text": "<реальный текст LLM>",
    "bboxes": [
      { "type": "signature", "coords": [x0, y0, x1, y1] },
      { "type": "stamp",     "coords": [x0, y0, x1, y1] }
    ]
  }
  ```

---

## Настройка и расширение

* **Новые типы документов**: добавьте шаблон в `prompts.json`.
* **Эффекты сканирования**: настройте `aug = A.Compose([...])` в `syntheticdocs.py`.
* **Многополосные документы**: расширьте `draw_pdf()`.
* **Параллельность**: используйте батчи vLLM или `asyncio`.

---

## Лицензия и отказ от ответственности

Код предоставляется «как есть» и предназначен **исключительно** для исследовательских целей.
Авторы не несут ответственности за неправильное использование.

```
```
