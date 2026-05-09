#!/bin/bash
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
python -m py_compile train_resnet.py dataset.py ssl_simclr.py resnet18/*.py
python - <<'PY'
import torch
from resnet18 import build_resnet
from ssl_simclr import SimCLR, nt_xent_loss

torch.set_num_threads(1)
for arch in ["resnet34", "resnet50"]:
    backbone = build_resnet(arch, num_classes=10).eval()
    feat_dim = backbone.feature_dim
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        y = backbone(x)
        f = backbone(x, return_features=True)
    assert y.shape == (2, 10), (arch, y.shape)
    assert f.shape == (2, feat_dim), (arch, f.shape)
    backbone.fc = torch.nn.Identity()
    model = SimCLR(backbone, feat_dim=feat_dim, proj_dim=128, proj_layers=3).eval()
    with torch.no_grad():
        z1 = model(x)
        z2 = model(x)
    assert z1.shape == (2, 128), (arch, z1.shape)
    loss = nt_xent_loss(z1, z2, temperature=0.2)
    assert torch.isfinite(loss), loss
print("smoke test ok")
PY
