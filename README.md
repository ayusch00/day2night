# SemGAN Day-to-Night Image Translation

Research code for unpaired day-to-night and night-to-day image translation with
semantic consistency. The repository contains the training and inference
pipeline used for the final thesis experiment; datasets and model weights are
kept outside Git.

## Method

The method combines three components:

```text
Cityscapes labels ──> segmentation pretraining ──> frozen generator encoder
                                                        │
unpaired day/night images ──> CycleGAN (G and F) <───────┘
                                  │
                                  └──> frozen DTBS segmenter
                                       └──> bidirectional semantic loss
```

1. A nine-residual-block segmentation network is pretrained on Cityscapes.
   Its best encoder weights initialize the day-to-night generator `G` and
   remain frozen during translation training. The reverse generator `F`
   starts from a random, trainable encoder.
2. Two generators learn day-to-night (`G`) and night-to-day (`F`) mappings.
   Two PatchGAN discriminators, least-squares adversarial loss, cycle
   consistency, identity preservation, and an image replay pool stabilize
   unpaired training.
3. A frozen DTBS Cityscapes-to-ACDC-night segmenter supplies a semantic
   consistency loss for both translation directions. The final experiment uses
   an L1 distance between class-probability maps.

## Final thesis configuration

The two versioned configurations reproduce the selected proportional runs:
`configs/segmentation.yaml` for the `seg_prop` encoder and
`configs/semgan.yaml` for the final `semgan_prop5k` translation. In both
training pipelines, the integer `resize: 572` preserves the aspect ratio
before a 512 x 512 crop.

| Component | Final setting |
| --- | --- |
| Translation data | 5,000 training + 1,000 held-out images per domain, balanced across five luminance quantiles |
| Training transform | resize shorter side to 572 px, random 512 x 512 crop, horizontal flip |
| Encoder pretraining | `seg_prop`; all 2,975 Cityscapes training images; proportional resize/crop; 9 residual blocks; 300 epochs; batch 8 |
| Generator | frozen transferred encoder for day-to-night only; 9 encoder residual blocks; no additional decoder residual blocks |
| Discriminator | four-layer PatchGAN, LSGAN objective |
| Cycle / identity loss | 10.0 / 5.0 |
| Semantic loss | DTBS, both directions, L1 probability loss, weight 1.0 |
| Translation schedule | 300 epochs, linear decay from epoch 150, batch 1, replay pool 50, AMP |
| Optimizers | Adam; generator 2e-4, discriminator 1e-4 |

## Repository layout

```text
configs/segmentation.yaml final encoder-pretraining settings
configs/semgan.yaml        final semantic CycleGAN settings
src/train_seg.py           encoder pretraining
src/train_cyclegan.py      CycleGAN + semantic-consistency training
src/apply_cyclegan.py      minimal translation inference CLI
src/apply_segnet.py        segmentation inference/visualization CLI
scripts/setup_env.sh       reproducible Python 3.10 environment setup
scripts/score_images.py    luminance metadata generation
scripts/prepare_dataset.py reproducible five-bucket split
data/README.md             dataset layout and BDD100K example
checkpoints/README.md      required checkpoint files
```

Runtime outputs are written to `experiments/` and `results/`; both are
ignored by Git.

## Requirements and environment

The tested environment is Python 3.10 with PyTorch 2.5.1, torchvision 0.20.1,
CUDA 12.1 wheels, MMCV 1.4.0, and the DTBS MMSegmentation 0.16.0 source tree.
Training requires an NVIDIA CUDA setup with sufficient memory for 512 x 512
crops.

To create a virtual environment named `semgan310`, install the pinned
requirements, clone DTBS, and check the imports:

```bash
VENV_DIR="$PWD/semgan310" ./scripts/setup_env.sh
source semgan310/bin/activate
```

For the conventional `.venv` name, simply run:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

The setup searches for a Python 3.10 interpreter. If it is installed through
Conda, pyenv, or a non-standard system path, pass it explicitly:

```bash
PYTHON_BIN=/absolute/path/to/python3.10 \
  VENV_DIR="$PWD/semgan310" \
  ./scripts/setup_env.sh
```

DTBS is not included in this Git repository. The setup script automatically
clones it under `third_party/DTBS` and checks out commit
`ea92f6910a1b36c12625a54789cdeb6a6e5dbff4`. It is kept as an external
dependency because its upstream repository does not declare a source license.
Verify an existing environment at any time with:

```bash
python scripts/check_environment.py
```

Run the repository regression tests after setup:

```bash
make test
```

They validate both final configs, deterministic dataset preparation, encoder
weight transfer, and core model tensor shapes without starting a training run.

## Data and checkpoints

Follow [data/README.md](data/README.md) to create:

```text
data/trainA  data/trainB  data/testA  data/testB
```

The same guide documents the exact luminance-balanced 5k split and a complete
BDD100K example. Encoder pretraining additionally requires the official
Cityscapes `leftImg8bit` and `gtFine` trees.

Follow [checkpoints/README.md](checkpoints/README.md) for the three expected
weight files:

```text
checkpoints/encoder_GE.pth
checkpoints/dtbs/cs2acdc_latest.pth
checkpoints/semgan_prop5k.pt
```

### Optional checkpoint bundle

As an alternative to obtaining the files individually, download the complete
checkpoint folder from [TU Berlin TubCloud](https://tubcloud.tu-berlin.de/s/CY7qWQ2XMgHmpqF).
Copy the downloaded `checkpoints/` folder to the repository root, or copy its
contents into the existing local `checkpoints/` directory. This download is
optional; the individual setup instructions remain available in
[checkpoints/README.md](checkpoints/README.md).

Only the first two are needed to start translation training. The complete
`semgan_prop5k.pt` file is used for final-run inference and already contains
both translation generators. The optional notebook reference
`checkpoints/lastchance_semgan_20260613_023510_latest.pt` is a separate
epoch-200, 1,500-image run and is not required for training.

## Training

Run every command from the repository root with the virtual environment
activated.

### 1. Pretrain the segmentation encoder

Single GPU:

```bash
python -m src.train_seg --config configs/segmentation.yaml
```

Four GPUs with DistributedDataParallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29503 \
  -m src.train_seg \
  --config configs/segmentation.yaml
```

The best validation model is written to
`experiments/seg_prop_<timestamp>/encoder_GE.pth` together with the full
segmentation model `segnet_full.pth`. The final thesis run used all 2,975
images in the official Cityscapes training split; `sample_n_train: 5000` was
an upper bound and did not subsample that split.

The encoder used by the final SemGAN run came exactly from
`experiments/seg_prop_20260626_115203/encoder_GE.pth`. Its 21 tensors match
the frozen encoder stored in generator `G` of the epoch-300 checkpoint.
Copy a reproduced encoder to the reusable path configured in `semgan.yaml`:

```bash
cp experiments/seg_prop_<timestamp>/encoder_GE.pth checkpoints/encoder_GE.pth
```

### 2. Train semantic CycleGAN

Single GPU:

```bash
python -m src.train_cyclegan --config configs/semgan.yaml
```

Four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29504 \
  -m src.train_cyclegan \
  --config configs/semgan.yaml
```

Choose a unique `--master_port` for every simultaneous `torchrun` job on the
same host. The global batch size equals `train.batch_size` from `semgan.yaml` times
the number of processes.

Resume a run with the same process count:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29504 \
  -m src.train_cyclegan \
  --config configs/semgan.yaml \
  --resume experiments/semgan_prop5k_<timestamp>/latest.pt
```

Each run stores `latest.pt`, the lowest-generator-loss `best.pt`, the
resolved config, and logs below `experiments/semgan_prop5k_<timestamp>/`.
Epoch checkpoints are disabled in the final configuration to avoid redundant
large files.

Equivalent shortcuts are available through `make train-seg`,
`make train-translation`, `make train-seg-ddp GPUS=4 MASTER_PORT=29503`, and
`make train-translation-ddp GPUS=4 MASTER_PORT=29504`.

## Inference

The CLI is the canonical inference path. The optional, output-free
[inference notebook](notebooks/inference_demo.ipynb) uses the same tested
loading and transformation helpers, selects ten random images with a fixed
seed, generates only those samples, resizes each generated image back to the
corresponding input resolution with bicubic interpolation, and displays every
input/output pair in one two-column comparison figure.

Day to night:

```bash
python -m src.apply_cyclegan \
  --config configs/semgan.yaml \
  --checkpoint checkpoints/semgan_prop5k.pt \
  --direction day2night \
  --input-dir /path/to/day-images \
  --output-dir results/inference
```

Night to day:

```bash
python -m src.apply_cyclegan \
  --config configs/semgan.yaml \
  --checkpoint checkpoints/semgan_prop5k.pt \
  --direction night2day \
  --input-dir /path/to/night-images \
  --output-dir results/inference
```

Generated files preserve their relative input paths below
`results/inference/<direction>/`. Inference from a complete CycleGAN
checkpoint does not load the encoder-pretraining or DTBS checkpoint.

To inspect the full segmentation model:

```bash
python -m src.apply_segnet \
  --config configs/segmentation.yaml \
  --checkpoint experiments/seg_prop_<timestamp>/segnet_full.pth \
  --input-dir /path/to/images \
  --output-dir results/segmentation
```

## Reproducibility notes

- Random seeds are stored in the config; data preparation additionally writes
  explicit selection manifests.
- The complete translation checkpoint embeds the resolved translation config.
- Dataset redistribution is intentionally avoided. Record the dataset versions,
  licenses, and generated manifests with any released model.
- CUDA kernels and multi-GPU scheduling can still introduce small numerical
  differences between runs.
- Large checkpoints should be attached to a release or archival record rather
  than committed to Git.

## External projects and datasets

This work builds on [CycleGAN](https://junyanz.github.io/CycleGAN/),
[DTBS](https://github.com/hf618/DTBS),
[Cityscapes](https://www.cityscapes-dataset.com/), and
[BDD100K](https://www.vis.xyz/bdd100k/). Cite and follow the terms of the
projects and datasets used in a reproduction.

## Citation and license

If you use this repository, cite the master's thesis using the metadata in
[`CITATION.cff`](CITATION.cff). The source code is released under the
[`MIT License`](LICENSE). External software, models, and datasets remain subject
to their own license and usage terms.
