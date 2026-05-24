# Masters diploma — NLP-пайплайн для научных статей

Приложение скачивает (или берёт локальные) PDF с arXiv, извлекает текст через
[GROBID](https://github.com/kermitt2/grobid), распознаёт сущности (NER) и
отношения (RE) на базе SciBERT, строит граф знаний в Neo4j и собирает метрики
производительности.

## Требования

- Python 3.11+
- Docker и Docker Compose (Neo4j + GROBID)
- GPU рекомендуется для обучения и инференса SciBERT (CPU возможен, но
  медленнее)

## Установка

```sh
# Клонировать репозиторий и перейти в каталог проекта
cd masters_diploma

# Виртуальное окружение (рекомендуется)
python -m venv .venv
source .venv/bin/activate # Активация окружения

# Зависимости
pip install -e .
# или: pip install -r requirements.txt
```

## Инфраструктура (Neo4j и GROBID)

Перед основным прогоном поднимите сервисы:

```sh
docker compose up -d
```

| Сервис | URL / порт | Примечание | 
| ------------- | ------------------------ | --------------------------------- | 
| Neo4j Browser | http://localhost:7474 | `NEO4J_AUTH=none` — пароль пустой | 
| Neo4j Bolt | `neo4j://localhost:7687` | используется пайплайном | 
| GROBID | http://localhost:8070 | health-check перед обработкой PDF |

Проверка GROBID: в браузере или
`curl -s -o /dev/null -w "%{http_code}" http://localhost:8070` — ожидается
`200`.

## Быстрый старт

1. Обучить модели на SciERC (один раз, если нет чекпоинтов в `artifacts/`):

   ```sh
   python main.py ner_re_learn
   ```

1. Положить PDF в `.cache/papers/<arxiv_id>/<arxiv_id>.pdf` **или** скачать с
   arXiv:

   ```sh
   python main.py run --download --categories math.SG math.SP --num-each 5
   ```

1. Запустить пайплайн по локальным статьям (режим по умолчанию):

   ```sh
   python main.py
   # то же самое:
   python main.py run
   ```

Результаты метрик: `.cache/metrics/<дата-время>/` (`metrics.jsonl`,
`summary.csv`, `figures/`).\
Логи: `.log/`.\
Чекпоинты моделей: `artifacts/ner`, `artifacts/re`.

## Команды CLI

Общий вид:

```sh
python main.py [COMMAND] [опции]
```

Если `COMMAND` не указана, выполняется **`run`**. Подкоманды можно писать через
дефис или подчёркивание (`ner_re_learn` ≡ `ner-re-learn`).

Справка:

```sh
python main.py --help
python main.py run --help
python main.py ner-re-learn --help
```

### `run` — основной пайплайн

Обработка статей: GROBID → NER/RE → анализ → Neo4j → метрики.

```sh
python main.py run [опции]
python main.py --download --categories cs.AI --num-each 10   # run по умолчанию
```

| Опция | По умолчанию | Описание | 
| -------------------- | ------------------------ | ------------------------------------ | 
| `--categories CAT …` | `math.SG` `math.SP` | Категории arXiv | 
| `--num-each N` | `20` | Статей на категорию при `--download` | 
| `--download` | — | Скачать PDF с arXiv перед обработкой | 
| `--papers-dir PATH` | `.cache/papers` | Каталог с локальными PDF | 
| `--grobid-url URL` | `http://localhost:8070` | Адрес GROBID |
| `--neo4j-uri URI` | `neo4j://localhost:7687` | Bolt URI | 
| `--neo4j-user` | `neo4j` | Пользователь | 
| `--neo4j-password` | _(пусто)_ | Пароль |
| `--ner-model PATH` | `artifacts/ner` | Чекпоинт NER | 
| `--re-model PATH` | `artifacts/re` | Чекпоинт RE |

Примеры:

```sh
# Только локальные PDF из .cache/papers
python main.py run

# Скачать и обработать
python main.py run --download --categories math.SG --num-each 3

# Свои чекпоинты
python main.py run --ner-model artifacts/ner --re-model artifacts/re
```

### `ner_re_learn` — обучение NER и RE на SciERC

Последовательно обучает обе модели на датасете `datasets/scierc/`.

```sh
python main.py ner_re_learn
python main.py ner_re_learn --epochs 5 --batch-size 4
```

Общие параметры обучения (для `train-ner`, `train-re`, `ner_re_learn`):

| Опция | По умолчанию | 
| ----------------- | ---------------------------------- | 
| `--train` | `datasets/scierc/train.json` | 
| `--dev` | `datasets/scierc/dev.json` | 
| `--model-name` | `allenai/scibert_scivocab_uncased` | 
| `--epochs` | `3` | 
| `--batch-size` | `8` | 
| `--learning-rate` | `2e-5` | 
| `--max-length` | `256` |

Для `ner_re_learn` дополнительно: `--ner-output-dir` (`artifacts/ner`),
`--re-output-dir` (`artifacts/re`).

Отдельное обучение:

```sh
python main.py train-ner --output-dir artifacts/ner
python main.py train-re --output-dir artifacts/re
```

### `rebuild-metrics` — пересборка отчётов по метрикам

Если уже есть `metrics.jsonl` от прошлого прогона, можно заново построить
`summary.csv` и графики без повторной обработки статей:

```sh
python main.py rebuild-metrics .cache/metrics/2026-05-09_14-30-00
```

## Структура каталогов

```
.cache/
  papers/          # PDF по arXiv ID
  metrics/         # прогоны метрик (timestamp)
.log/              # логи приложения
artifacts/
  ner/             # чекпоинт NER
  re/              # чекпоинт RE
datasets/scierc/   # train.json, dev.json, test.json
src/               # исходный код
```

## Дополнительные скрипты

| Скрипт | Назначение | 
| ------------------------------ | ----------------------------------------------------- | 
| `evaluate_baseline.py` | Оценка NER/RE на тестовом сплите SciERC | 
| `analyze_ner.py` | Анализ корпуса и метрик NER | 
| `train_ner.py` / `train_re.py` | Обучение (дублируют `main.py train-ner` / `train-re`) |

## Зависимости

Основные пакеты (см. `pyproject.toml` / `requirements.txt`):

- arxiv, lxml, spacy, spacy-transformers
- transformers, torch, accelerate
- neo4j, psutil

Для метрик GPU (опционально): `nvidia-ml-py` или `nvidia-smi` в PATH.
