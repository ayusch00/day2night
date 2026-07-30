# Data

No dataset images are distributed with this repository. Download and use each
dataset under its own license or terms of access. The contents of this directory,
except for this file, are ignored by Git.

## Translation dataset layout

The translation configuration in `configs/semgan.yaml` expects:

```text
data/
├── trainA/       # day-domain training images
├── trainB/       # night-domain training images
├── testA/        # held-out day images
├── testB/        # held-out night images
└── manifests/    # generated split manifests (optional)
```

The domains are unpaired: matching filenames or aligned day/night image pairs are
not required.

## Reproduce the luminance-balanced 5k split

Place the complete source pools in two arbitrary directories. First compute the
same image statistics that were used for the thesis experiment:

```bash
python scripts/score_images.py \
  --input-dir /path/to/all-day-images \
  --output data/metadata/day_scores.csv \
  --domain day

python scripts/score_images.py \
  --input-dir /path/to/all-night-images \
  --output data/metadata/night_scores.csv \
  --domain night
```

Then create the four configured folders:

```bash
python scripts/prepare_dataset.py \
  --day-dir /path/to/all-day-images \
  --night-dir /path/to/all-night-images \
  --day-csv data/metadata/day_scores.csv \
  --night-csv data/metadata/night_scores.csv \
  --out-dir data
```

The defaults reproduce the final split procedure:

- convert each image to RGB and resize it to 100 x 100 pixels;
- calculate luminance as `0.299 R + 0.587 G + 0.114 B`;
- min-max normalize mean luminance separately within each domain;
- sort each domain and divide it into five equally sized quantile buckets;
- draw 1,000 training and 200 test images per bucket with seed `6311`.

The result contains 5,000 training and 1,000 held-out images per domain.
Corrupt images and rows with missing files are excluded. The generated CSV
manifests record the original filename, score, bucket, and destination filename.
Use `--mode symlink` with `prepare_dataset.py` to avoid duplicating source
images on the same machine.

## BDD100K example

After obtaining BDD100K from the [official project
page](https://www.vis.xyz/bdd100k/), separate its training images using the
official `attributes.timeofday` labels:

```bash
python scripts/prepare_bdd100k_domains.py \
  --images /path/to/bdd100k/images/100k/train \
  --labels /path/to/bdd100k/labels/bdd100k_labels_images_train.json \
  --out-dir data/bdd100k_domains \
  --limit 6000
```

Score the two generated source folders and pass their CSV files to
`prepare_dataset.py`:

```bash
python scripts/score_images.py \
  --input-dir data/bdd100k_domains/day \
  --output data/metadata/bdd_day_scores.csv \
  --domain day

python scripts/score_images.py \
  --input-dir data/bdd100k_domains/night \
  --output data/metadata/bdd_night_scores.csv \
  --domain night

python scripts/prepare_dataset.py \
  --day-dir data/bdd100k_domains/day \
  --night-dir data/bdd100k_domains/night \
  --day-csv data/metadata/bdd_day_scores.csv \
  --night-csv data/metadata/bdd_night_scores.csv \
  --out-dir data
```

This is a reusable example of the same preparation method. It does not claim to
reconstruct the exact mixed-source image pool used for the final thesis run.

## Cityscapes layout for encoder pretraining

The `configs/segmentation.yaml` configuration expects the official
Cityscapes archives extracted as follows:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

Cityscapes is available through its [official download
portal](https://www.cityscapes-dataset.com/downloads/). Keep the directory names
or adjust the four Cityscapes paths in the configuration.
