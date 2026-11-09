# MLOps Lab 3 — Toxicity Classifier Training

Дообучение модели `cointegrated/rubert-tiny2` на подмножестве датасета
`AlexSham/Toxic_Russian_Comments` с использованием Hydra, DVC и MLflow.

## Стек

- **Hydra** — управление конфигами экспериментов (`config/config.yaml`).
- **DVC** — версионирование данных и чекпоинтов.
- **MLflow** — трекинг экспериментов (параметры, метрики, артефакты).
- **Transformers + PyTorch** — обучение.

## Установка

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Данные

Данные версионируются через DVC. Чтобы получить их:

```bash
dvc pull
```

Если remote не настроен — добавь локальный:

```bash
dvc remote add -d local C:\dvc_storage
dvc pull
```

Пересоздать подмножество из исходного датасета HuggingFace:

```bash
python prepare_data.py
dvc add data/train_subset.csv
dvc add data/val_subset.csv
dvc push
```

## Обучение

Базовый запуск (параметры из `config/config.yaml`):

```bash
python train.py
```

С переопределением гиперпараметров через CLI:

```bash
python train.py training.learning_rate=1e-4 training.batch_size=32
python train.py training.learning_rate=5e-5 training.epochs=5
```

## Трекинг экспериментов

```bash
mlflow ui
```

Открой http://127.0.0.1:5000 — увидишь все запуски эксперимента `toxicity-rubert`.

## Лучший запуск

Лучший результат: **F1 = 0.9155, Accuracy = 0.915** при:

- `learning_rate = 5e-5`
- `batch_size = 16`
- `epochs = 5`
- `seed = 42`

Воспроизвести:

```bash
python train.py training.learning_rate=5e-5 training.epochs=5
```

## Воспроизводимость

Для полной воспроизводимости зафиксировано:

1. **Seed** — `training.seed=42` в конфиге, применяется к `random`, `numpy`, `torch`.
2. **Версии зависимостей** — `requirements.txt`.
3. **Версия данных** — DVC-хеши в `data/*.dvc`.
4. **Версия кода** — git commit.
5. **Конфиг** — логируется в MLflow как артефакт.

## Структура проекта

```
.
├── config/
│   └── config.yaml          # Hydra-конфиг эксперимента
├── data/
│   ├── train_subset.csv.dvc # метаданные DVC (данные — в remote)
│   └── val_subset.csv.dvc
├── prepare_data.py          # скрипт подготовки подмножества
├── train.py                 # обучение + MLflow логирование
├── requirements.txt
└── README.md
```