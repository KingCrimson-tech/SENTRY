.PHONY: setup data aug tts chan train ood eval app check

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

# 3 real shards + 3 fake shards (~2.8 GB total, balanced)
data:
	.venv/bin/python -m src.data --indices 0,1,2 --windows 2
	.venv/bin/python -m src.data --indices 10,11,12 --windows 2

aug:
	.venv/bin/python -m src.augment --per-shard 600 --variants 3

tts:
	.venv/bin/python -m src.make_samples --generate 200 --cache

chan:
	.venv/bin/python -m src.channel_data --collect --cache

train:
	.venv/bin/python -m src.train
	.venv/bin/python -m src.ood --build

ood:
	.venv/bin/python -m src.ood --build

eval:
	.venv/bin/python -m src.evaluate

app:
	.venv/bin/python app.py

check:
	.venv/bin/python scripts/demo_check.py
