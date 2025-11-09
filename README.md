python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Uses the cu121 extra index
pip install -r requirements.txt

## CycleGAN Day↔Night Training

1. Legen Sie Ihre Tages- und Nachtbilder unter `data/day` bzw. `data/night` ab (kann in `configs/cyclegan.yaml` angepasst werden).
2. Legen Sie die trainierten Encoder-Gewichte aus der Segmentierung (Standard: `experiments/seg_cityscapes_1k_encoder_GE.pth`) bereit oder passen Sie den Pfad unter `model.generator.encoder_checkpoint` an.
3. Optional: Passen Sie Hyperparameter, Pfade oder Augmentationen in `configs/cyclegan.yaml` an.
4. Starten Sie das Training:

```bash
python src/train_cyclegan.py --config configs/cyclegan.yaml
```

Checkpoints werden unter `experiments/` abgelegt (inkl. Generatoren G/F und Discriminatoren Ds/Dt).
