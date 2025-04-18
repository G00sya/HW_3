# HW_3
Cool repository with 1st ML homework

# Окружение
Сперва установите `poetry`
```bash
pip install poetry
```

Для создания окружения и установки всех нужных пакетов запустите
```bash
make install
```

Для установки хуков
```bash
make hooks
```

Для запуска линтеров
```bash
make lint
```

Для запуска тестов
```bash
make test
```

Для обновления и фиксации зависимостей
```bash
poetry lock
```

Для активации виртуального окружения скопируйте то, что выдает эта команда и запустите скопированную команду
```bash
poetry env use python3.12
poetry env activate
```

Для установки данных для обучения
```bash
mkdir -p data/raw
wget -O data/raw/news.zip -qq --no-check-certificate "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab"
unzip data/raw/news.zip -d data/raw
rm data/raw/news.zip


Для установки предобученного эмбединга
```bash
mkdir -p ./embeddings
wget -P ./embeddings https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
unzip ./embeddings/glove.6B.zip -d ./embeddings/ && rm ./embeddings/glove.6B.zip
```

# Структура проекта

```
project_root/
├── .git/                    # Git repository (managed by Git)
├── .dvc/                    # DVC metadata directory (managed by DVC)
├── .wandb/                  # Weights & Biases local data (often hidden)
├── src/                     # Source code directory
│   ├── __init__.py
│   ├── model/               # Transformer model src code
│   │   ├── __init__.py
│   │   ├── *.py             # Class definitions, architectures
│   │   ├── *.py             # Helper functions for model loading, saving, etc.
│   ├── data/                # Data processing and loading
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Loads the database
│   │   ├── preprocessing.py # Data cleaning, normalization
│   ├── features/            # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_engineering.py  # Feature extraction functions
│   ├── utils/               # Utility functions and modules
│   │   ├── __init__.py
│   │   ├── *.py
│   ├── main.py
│
├── data/                    # Raw data and processed data
│   ├── raw/                 # Original, unprocessed data (DVC tracked)
│   │   ├── news.csv
│   ├── processed/           # Processed data (DVC tracked)
│   │   ├── processed_data.csv
│
├── model/                   # Trained model weights and configurations (DVC tracked)
│   │   ├── model.pt         # PyTorch weights (or .pkl for scikit-learn, etc.)
│   │   ├── config.json      # Model configuration (hyperparameters)
│
├── notebooks/               # Jupyter notebooks for demonstration
│
├── tests/                   # Tests directory
│   ├── __init__.py
│
```
