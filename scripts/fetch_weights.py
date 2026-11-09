"""Download the pretrained weights used by the experiments into models/.

- DINOv3 ViT-L/16 (SAT-493M) and ViT-B/16 (LVD-1689M) via timm, stored as fp16 safetensors in models/dinov3/
  (DINOv3 licence: check the model cards before any commercial use).
- DOFA ViT-B/16 weights from Hugging Face (earthflow/DOFA, CC-BY-4.0) into models/dofa/.
- AnySat is fetched by torch.hub on first use; ResNet-34 (ImageNet) by timm on first use.
Usage: PYTHONPATH=src python scripts/fetch_weights.py
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcrs_ai import paths  # noqa: E402,F401  (redirects caches into the project)

import timm  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from pcrs_ai.dofa import WEIGHTS_URL  # noqa: E402
from pcrs_ai.models import ENCODERS  # noqa: E402

out = paths.MODELS / "dinov3"
out.mkdir(parents=True, exist_ok=True)
for arch in ENCODERS.values():
    target = out / f"{arch}.fp16.safetensors"
    if not target.exists():
        m = timm.create_model(arch, pretrained=True, num_classes=0)
        save_file({k: v.half().contiguous() for k, v in m.state_dict().items()}, str(target))
        print("saved", target)
dofa = paths.MODELS / "dofa" / "DOFA_ViT_base_e100.pth"
if not dofa.exists():
    dofa.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WEIGHTS_URL, dofa)
    print("saved", dofa)
