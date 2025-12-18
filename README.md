python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Uses the cu121 extra index
pip install -r requirements.txt

## CycleGAN Day↔Night Training

1. Legen Sie Ihre Tages- und Nachtbilder unter `data/day` bzw. `data/night` ab (kann in `configs/cyclegan.yaml` angepasst werden). (1000 bilder each!)
2. Legen Sie die trainierten Encoder-Gewichte aus der Segmentierung bereit. Setzen Sie `model.generator.encoder_checkpoint` auf
   - `latest` (Standard, nimmt den jüngsten `experiments/seg_*/encoder_GE.pth`),
   - `latest:<glob>` für eigene Muster (z. B. `latest:seg_cityscapes_*`),
   - oder einen konkreten Pfad/Ordner mit `encoder_GE.pth`.
3. Optional: Passen Sie Hyperparameter, Pfade oder Augmentationen in `configs/cyclegan.yaml` an.
4. Starten Sie das Training (200 Epochen, davon 100 konstant, danach lineare LR-Absenkung):

   - Für den Einzelprozessor-Workflow:

     ```bash
     python -m src.train_cyclegan --config configs/cyclegan.yaml
     ```

   - Für Multi-GPU mit DDP (empfohlen auf 8 GPUs, Batch 1/Prozess):

     ```bash
     torchrun --nproc_per_node=8 -m src.train_cyclegan --config configs/cyclegan.yaml
     ```

Checkpoints werden unter `experiments/` abgelegt (inkl. Generatoren G/F und Discriminatoren Ds/Dt).

- **Stabilitäts-Tuning:** Standardmäßig läuft TTUR (G = 2e‑4, D = 1e‑4), die Discriminatoren können optional per SpectralNorm verstärkt werden (`model.discriminator.use_spectral_norm`) und ein Replay-Buffer (`train.image_pool_size`, default 50) glättet das D-Training.
- **Sky-Identity-Loss (optional):** Setze `sky_loss.enabled: true` in `configs/cyclegan.yaml`, trage `class_ids` (z. B. Cityscapes sky=10, vegetation=8). `seg_checkpoint: null`/`latest` sucht automatisch das jüngste `experiments/seg_*/segnet_full.pth` (oder setze explizit einen Pfad). Der Loss bremst neue Punktlichter im Himmel/Baum-Bereich für Day→Night, mit optionaler Glättung (`tv_weight`).

### Segmentierungs‑Pretraining (Encoder)

Die CycleGAN‑Generatoren nutzen einen Encoder, der zuvor auf Cityscapes segmentiert wurde. Trainiert ihn mit:

```bash
python -m src.train_seg --config configs/seg.yaml
```

Für Multi‑GPU‑Training empfiehlt sich DDP via `torchrun` (analog zu CycleGAN). Das Skript erkennt `WORLD_SIZE` automatisch:

```bash
torchrun --nproc_per_node=8 -m src.train_seg --config configs/seg.yaml
```

Der Encoder-Checkpoint landet standardmäßig unter `experiments/<seg_run>/encoder_GE.pth`. `configs/cyclegan.yaml` greift automatisch auf den jüngsten `seg_*`-Lauf zu (`encoder_checkpoint: latest`), kann bei Bedarf aber weiterhin auf einen festen Pfad zeigen.

**Andere Datensätze:** Über `data.dataset` lässt sich der Seg-Loader umschalten (`"cityscapes"` oder `"bdd100k"`). Für BDD müssen zudem passende `train_images`/`train_masks` angegeben werden; optional begrenzt `data.extensions` die erlaubten Dateiendungen.

## BDD100K Subset (optional)

Erstelle ein kleines CycleGAN-Trainingsset aus BDD100K:

```bash
python data/smallset_bdd.py --root data/bdd100k --n 1000
```

## CycleGAN Inference

Einfachere Eingabe‑/Ausgabe-Pfade und die Anwendung eines gespeicherten Generators bietet das neue Skript `src.apply_cyclegan`.

1. Legt eure Eingabebilder in ein beliebiges Verzeichnis (z. B. `data/night2day/night_to_day/testA` für Tagesbilder oder `testB` für Nachtbilder).
2. Gebt optional den Checkpoint an, sonst wird automatisch das jüngste `epoch_*.pt` aus `experiments/{exp_name}_*/` geladen.
3. Führt z. B. aus:

   ```bash
   python -m src.apply_cyclegan \
     --direction day2night \
     --input-dir data/night2day/night_to_day/testA \
     --checkpoint experiments/cyclegan_day2night_20251110_163433/epoch_0120.pt \
     --output-dir results/cyclegan_inference
   ```

   Die generierten Bilder landen unter `results/cyclegan_inference/day2night` (bzw. `.../night2day` bei der umgekehrten Richtung).

Weitere Optionen:

- `--config`: Pfad zur Konfigurationsdatei (Standard `configs/cyclegan.yaml`).
- `--extensions`: Erweiterungen, die durchsucht werden sollen (Standard aus der Konfiguration oder `jpg/jpeg/png/...`).
- Helligkeit steuern: In `configs/cyclegan.yaml` unter `inference` kannst du `brightness_gain` (linear, z. B. 1.15) und `output_gamma` (Gamma-Korrektur, z. B. 1.1) setzen, um Day→Night‑Ergebnisse aufzuhellen. Beide sind standardmäßig 1.0 (keine Änderung).

Der angegebene `--root` sollte direkt die Unterordner `images/` und `labels/` enthalten. Liegen keine `labels/bdd100k_labels_images_*.json` (mit `attributes.timeofday`) vor, kannst du mit `--heuristic` eine simple Helligkeits-Heuristik nutzen (optional `--threshold`, z. B. `0.5`). Mit `--copy` anstelle von Symlinks werden echte Dateien erzeugt.

### Test-Split erstellen

Sobald `trainA/trainB` stehen, kannst du über den Val-Split (oder bisher ungenutzte Train-Bilder) einen disjunkten Test-Satz erzeugen:

```bash
python data/make_test_split_bdd.py --root data/bdd100k --n 1000
```

Das Skript liest `labels/bdd100k_labels_images_{train,val}.json`, vermeidet Überschneidungen mit bestehenden `cyclegan/train*`, bevorzugt `val` und erstellt `cyclegan/testA` (Tag) und `testB` (Nacht). `--copy` erzwingt Kopien statt Symlinks.

Kurze Torchrun-Befehle:

- SegNet DDP: `torchrun --nproc_per_node=8 -m src.train_seg --config configs/seg.yaml`
- CycleGAN DDP: `torchrun --nproc_per_node=8 -m src.train_cyclegan --config configs/cyclegan.yaml`
torchrun --nproc_per_node=8 -m src.train_seg --config configs/seg.yaml
torchrun --nproc_per_node=8 -m src.train_cyclegan --config configs/cyclegan.yaml


torchrun --nproc_per_node=4 -m src.train_cyclegan --config configs/cyclegan.yaml --resume experiments/cyclegan_day2night_20251204_184245/epoch_0020.pt

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=4 -m src.train_seg --config configs/seg.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m src.train_cyclegan --config configs/cyclegan.yaml experiments/cyclegan_day2night_20251207_185813/epoch_0120.pt

torchrun --nproc_per_node=4 --master_port=29503 -m src.train_cyclegan --config configs/cyclegan.yaml/tmp/wait_for_gpu\ copy.sh.


torchrun --nproc_per_node=4 --master_port=29503 -m src.train_seg --config configs/seg.yaml
torchrun --nproc_per_node=4 --master_port=29503 -m src.train_cyclegan --config configs/cyclegan.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=29503 -m src.train_cyclegan --config "configs/cyclegan snow.yaml" 