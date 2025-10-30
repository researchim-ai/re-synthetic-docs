# SyntheticDocs

![image](https://github.com/user-attachments/assets/67292497-940a-4aa2-aa79-4bfc6d62e517)


**Набор инструментов для генерации синтетических документов для формирования датасетов и обучения Vision-Language моделей (VLM) на задачах понимания документов.**

> ❗️ **Только для исследовательских целей:** этот проект предназначен исключительно для научных исследований и экспериментов по обучению и оценке VLM. Сгенерированные документы **не должны** использоваться в реальных или производственных системах, а также ни в каких мошеннических или обманных целях.

---

## Зачем это нужно

Для многих задач VLM (распознавание компоновки документа, детекция подписей и печатей, OCR + привязка к layout) не хватает больших размеченных датасетов. **SyntheticDocs** автоматически генерирует:

- **Изображения «сканов»** документов (`.png`, 300 DPI)  
- **Оригинальные PDF** (`.pdf`)  
- **Метаданные** (`.json`) с:
  - Текстовым содержимым (непосредственно из LLM)
  - Bounding-box’ами для каждой подписи и печати
  - Типом документа и полями промпта

Эти три артефакта образуют полноценную обучающую выборку, позволяя VLM учиться и на визуальном, и на текстовом уровне одновременно.

---

## Ключевые возможности

- **Генерация текста LLM-моделью** через локальный vLLM + модели Hugging Face  
- **Шаблоны промптов** в `prompts.json` для разных жанров документов (служебная записка, счёт-фактура, договор и др.)  
- **Кириллический шрифт** для реалистичного русского текста в PDF  
- **Наложение подписей и печатей** (PNG-оверлеи) с автo-масштабированием и сохранением прозрачности  
- **Эффекты сканирования**: перспектива, размытие, JPEG-сжатие, шум (Albumentations)  
- **Выходные файлы**:
  - `.pdf` — исходный PDF
  - `.png` — синтетический «скан» (300 DPI)
  - `.json` — метаданные (bounding-box’ы + текст)

---

## Структура проекта

```text
re-synthetic-docs/
├── assets/
│   ├── signatures/
│   │   ├── fonts/      # .ttf-шрифты для генерации подписей
│   │   └── png/        # готовые PNG-подписи (α-канал)
│   └── stamps/
│       ├── fonts/      # .ttf-шрифты для генерации печатей
│       └── png/        # готовые PNG-печати (α-канал)
├── prompts.json        # JSON-шаблоны промптов по типам документов
├── make_signatures.py  # скрипт генерации подписи из шрифтов
├── make_stamps.py      # скрипт генерации круглых печатей
├── syntheticdocs.py    # основной генератор: LLM → PDF → PNG → метаданные
└── requirements.txt    # Python-зависимости

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

> **Примечания:**
>
> * Требуется Python 3.10+
> * Для GPU-ускорения установите совместимую сборку `torch` + CUDA **до** установки `vllm`.
> * На Linux может понадобиться `poppler-utils` для конвертации PDF→PNG:
>
>   ```bash
>   sudo apt update && sudo apt install -y poppler-utils
>   ```

---

## Подготовка ассетов

1. **Шрифты**
   Положите хотя бы один `.ttf` с поддержкой кириллицы в
   `assets/signatures/fonts/` и `assets/stamps/fonts/`.

2. **Генерация подписей** (опционально)

   С использованием конфига (рекомендуется):

   ```bash
   python make_signatures.py --config config.example.json
   ```

   Быстрый запуск без файла (CLI перекрывает конфиг):

   ```bash
   python make_signatures.py --num 200
   ```

   Результат: PNG-подписи появятся в `assets/signatures/png/` (или каталоге из конфига).

3. **Генерация печатей** (опционально)

   ```bash
   python make_stamps.py
   ```

   — создаст \~30 PNG-печатей в `assets/stamps/png/`.

---

## Генерация датасета

Запустите основную программу:

```bash
python syntheticdocs.py --config config.example.json
```

Все параметры берутся из `config.example.json`:
* `llm.*` — режим и параметры LLM (включая OpenAI-совместимый API)
* `generator.*` — шрифт, пути к подписям/печатьам, `out_dir`, количество `num`, `seed`

**Структура выходной папки:**

```text
out/
├── <uuid1>.pdf
├── <uuid1>.png
├── <uuid1>.json
├── <uuid2>.pdf
├── <uuid2>.png
├── <uuid2>.json
└── …
```

* `<uuid>.pdf` — PDF документа
* `<uuid>.png` — синтетический «скан» (300 DPI)
* `<uuid>.json` — метаданные:

  ```jsonc
  {
    "id": "<uuid>",
    "type": "<document_type>",
    "text": "<raw LLM text>",
    "bboxes": [
      { "type": "signature", "coords": [x0, y0, x1, y1] },
      { "type": "stamp",     "coords": [x0, y0, x1, y1] }
    ]
  }
  ```

---

## Конфигурация через JSON (LLM и подписи)

Вы можете передать конфигурацию через JSON-файл (CLI-параметры перекрывают значения из конфига):

```json
{
  "llm": {
    "mode": "openai",
    "base_url": "http://localhost:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "model": "gpt-4o-mini",
    "max_tokens": 512,
    "temperature": 0.7
  },
  "signatures": {
    "fonts_dir": "assets/signatures/fonts",
    "output_dir": "assets/signatures/png",
    "num": 500,
    "seed": 123,
    "locale": "auto",
    "font_size": { "min": 48, "max": 72 },
    "padding": 20,
    "rotation": { "min": -10, "max": 10 },
    "border": { "min": 0, "max": 10 },
    "blur": { "min": 0.0, "max": 0.8 }
  }
}
```

Запуск с конфигом:

```bash
python syntheticdocs.py --config config.example.json
```

Альтернатива без файла (CLI перекрывает конфиг, опционально):

```bash
python syntheticdocs.py \
  --llm-mode openai \
  --openai-base-url http://localhost:8000/v1 \
  --openai-api-key $OPENAI_API_KEY \
  --model gpt-4o-mini \
  --font assets/signatures/fonts/YourCyrillicFont.ttf 
```

Параметры `--max-tokens` и `--temperature` работают в обоих режимах.

---

## Расширение и настройка

* **Новые типы документов**: добавьте шаблон в `prompts.json`.
* **Настройка эффектов сканирования**: отредактируйте Albumentations pipeline в `syntheticdocs.py`.
* **Мультстраничность**: расширьте функцию `draw_pdf()`.
* **Параллельная генерация**: используйте батчи vLLM или `asyncio` для ускорения.

---

## Отказ от ответственности

Код предоставляется «как есть» и предназначен **исключительно** для исследовательских и учебных целей.
Авторы не несут ответственности за неправильное или злоупотребление данным инструментом.

```
```
