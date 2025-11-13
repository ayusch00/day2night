python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Uses the cu121 extra index
pip install -r requirements.txt

## CycleGAN Day↔Night Training

1. Legen Sie Ihre Tages- und Nachtbilder unter `data/day` bzw. `data/night` ab (kann in `configs/cyclegan.yaml` angepasst werden). (1000 bilder each!)
2. Legen Sie die trainierten Encoder-Gewichte aus der Segmentierung (Standard: `experiments/seg_cityscapes_1k_encoder_GE.pth`) bereit oder passen Sie den Pfad unter `model.generator.encoder_checkpoint` an.
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

Der angegebene `--root` sollte direkt die Unterordner `images/` und `labels/` enthalten. Liegen keine `labels/bdd100k_labels_images_*.json` (mit `attributes.timeofday`) vor, kannst du mit `--heuristic` eine simple Helligkeits-Heuristik nutzen (optional `--threshold`, z. B. `0.5`). Mit `--copy` anstelle von Symlinks werden echte Dateien erzeugt.

### Test-Split erstellen

Sobald `trainA/trainB` stehen, kannst du über den Val-Split (oder bisher ungenutzte Train-Bilder) einen disjunkten Test-Satz erzeugen:

```bash
python data/make_test_split_bdd.py --root data/bdd100k --n 1000
```

Das Skript liest `labels/bdd100k_labels_images_{train,val}.json`, vermeidet Überschneidungen mit bestehenden `cyclegan/train*`, bevorzugt `val` und erstellt `cyclegan/testA` (Tag) und `testB` (Nacht). `--copy` erzwingt Kopien statt Symlinks.
