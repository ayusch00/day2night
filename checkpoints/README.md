# Checkpoints

Model weights are intentionally not stored in Git. The two public configurations
use the following local files:

```text
checkpoints/
├── encoder_GE.pth
├── semgan_prop5k.pt
├── lastchance_semgan_20260613_023510_latest.pt  # optional notebook reference
└── dtbs/
    └── cs2acdc_latest.pth
```

## Optional complete download

The complete checkpoint folder is available from
[TU Berlin TubCloud](https://tubcloud.tu-berlin.de/s/CY7qWQ2XMgHmpqF).
After downloading it through the TubCloud web interface, place the
`checkpoints/` folder in the repository root. If the folder already exists,
copy the downloaded contents into it. The resulting paths should match the
layout above.

This bundle is optional. The sections below document the individual files and
their provenance.

## `encoder_GE.pth`

This is the best frozen generator encoder produced by the Cityscapes
`seg_prop` pretraining stage configured in `configs/segmentation.yaml`.
The final SemGAN run loaded
`experiments/seg_prop_20260626_115203/encoder_GE.pth`; the retained local copy
has SHA-256
`8b4a1f28caf458139f6a2d1a1d10dac31bedfb160ff2efcf12e7911d663ded37`
and all 21 encoder tensors exactly match generator `G` in the final checkpoint.
After reproduction, copy the new artifact from the created run directory:

```bash
cp experiments/seg_prop_<timestamp>/encoder_GE.pth checkpoints/encoder_GE.pth
```

## `dtbs/cs2acdc_latest.pth`

The semantic consistency loss uses the pretrained Cityscapes-to-ACDC-night
model from [DTBS](https://github.com/hf618/DTBS). Download the checkpoint from
the [upstream checkpoint
link](https://drive.google.com/file/d/1pi9sZmpUs8Nz5-nVu0Mt-itZkSj2xfa7/view?usp=sharing)
and place it at:

```text
checkpoints/dtbs/cs2acdc_latest.pth
```

The matching model configuration is supplied by the pinned DTBS checkout at
`third_party/DTBS/work_dirs/acdc/acdc_pretrained.json`. Run
`scripts/setup_env.sh` to obtain that exact checkout.

## `lastchance_semgan_20260613_023510_latest.pt`

This optional checkpoint is the default reference used by
`notebooks/inference_demo.ipynb`. It is `latest.pt` from
`lastchance_semgan_20260613_023510`: an epoch-200 run trained with 1,500
images per domain. It is separate from the final proportional 5k thesis
checkpoint and is not needed for training.

## `semgan_prop5k.pt`

This is a complete translation checkpoint. It contains both generators,
both discriminators, optimizer state, the epoch, and the resolved training
configuration. The reported proportional 5k thesis figure used `latest.pt`
from epoch 300, so prepare the public inference artifact with:

```bash
cp experiments/semgan_prop5k_<timestamp>/latest.pt checkpoints/semgan_prop5k.pt
```

The run also produced `best.pt`, selected by the lowest training
`g_total` (epoch 28). That criterion is not a held-out perceptual quality
metric and was not the checkpoint recorded for the final 5k figure.

Inference uses generator `G` for day-to-night and generator `F` for
night-to-day. Once this complete checkpoint exists, inference does not also
require `encoder_GE.pth` or the DTBS checkpoint.
