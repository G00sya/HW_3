SHELL := /bin/sh

install:
	poetry install

test:
	poetry run pytest tests

lint:
	poetry run pre-commit run --show-diff-on-failure --color=always --all-files

hooks:
	poetry run pre-commit install --install-hooks

install_pretrained_embeddings:
	mkdir -p ./embeddings
	wget -P ./embeddings https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

install_dataset:
	mkdir -p data/raw
	curl -L -o data/raw/news.zip "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab"
	unzip data/raw/news.zip -d data/raw
	rm data/raw/news.zip
