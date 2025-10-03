PY=python
DATE=$(shell date +"%Y%m%d-%H%M%S")
EXPDIR=experiments/$(DATE)


.PHONY: env train-seg train-gan eval-fid demo


env:
@echo "Using system env. Ensure PyTorch+CUDA installed."


train-seg:
mkdir -p $(EXPDIR)
$(PY) -m src.train_seg --config configs/seg.yaml --out $(EXPDIR)


train-gan:
mkdir -p $(EXPDIR)
$(PY) -m src.train_gan --config configs/gan.yaml --out $(EXPDIR)


eval-fid:
$(PY) -m src.eval_fid --real data/bdd/night/ --fake results/images/


demo:
$(PY) -c "print('Run notebooks/main_pipeline.ipynb for the full E2E demo.')"