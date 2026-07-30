from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DTBS_ROOT = REPO_ROOT / "third_party" / "DTBS"


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 is required, found {sys.version.split()[0]}")
    if not DTBS_ROOT.is_dir():
        raise FileNotFoundError(
            f"DTBS not found at {DTBS_ROOT}. Run scripts/setup_env.sh first."
        )

    sys.path.insert(0, str(DTBS_ROOT))

    import mmcv  # type: ignore
    import mmseg  # type: ignore
    import timm
    import torch
    import torchvision
    from mmseg.models import build_segmentor  # type: ignore

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "mmcv": mmcv.__version__,
        "dtbs_mmseg": mmseg.__version__,
        "timm": timm.__version__,
    }
    for name, version in versions.items():
        print(f"{name}: {version}")
    print(f"DTBS: {DTBS_ROOT}")
    print(f"MMSeg builder: {build_segmentor.__module__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
