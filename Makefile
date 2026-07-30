PYTHON ?= python
SEG_CONFIG ?= configs/segmentation.yaml
SEMGAN_CONFIG ?= configs/semgan.yaml
GPUS ?= 1
MASTER_PORT ?= 29503
CHECKPOINT ?= checkpoints/semgan_prop5k.pt
DIRECTION ?= day2night
INPUT_DIR ?= data/testA
OUTPUT_DIR ?= results/inference

.PHONY: setup check test train-seg train-seg-ddp train-translation train-translation-ddp infer

setup:
	VENV_DIR="$(CURDIR)/semgan310" ./scripts/setup_env.sh

check:
	$(PYTHON) scripts/check_environment.py

test:
	$(PYTHON) -m unittest discover -s tests -v

train-seg:
	$(PYTHON) -m src.train_seg --config $(SEG_CONFIG)

train-seg-ddp:
	torchrun --nproc_per_node=$(GPUS) --master_port=$(MASTER_PORT) -m src.train_seg --config $(SEG_CONFIG)

train-translation:
	$(PYTHON) -m src.train_cyclegan --config $(SEMGAN_CONFIG)

train-translation-ddp:
	torchrun --nproc_per_node=$(GPUS) --master_port=$(MASTER_PORT) -m src.train_cyclegan --config $(SEMGAN_CONFIG)

infer:
	$(PYTHON) -m src.apply_cyclegan --config $(SEMGAN_CONFIG) --checkpoint $(CHECKPOINT) --direction $(DIRECTION) --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)
