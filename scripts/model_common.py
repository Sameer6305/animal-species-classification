"""Shared helpers for Phase-0 scripts and tests (paths, manifest, torch model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet50_best.pt"

ONNX_PATH = MODELS_DIR / "resnet50.onnx"
ONNX_INT8_PATH = MODELS_DIR / "resnet50.int8.onnx"
ONNX_FP16_PATH = MODELS_DIR / "resnet50.fp16.onnx"
CLASSES_PATH = MODELS_DIR / "classes.json"
FC_NPZ_PATH = MODELS_DIR / "resnet50_fc.npz"
FC_JSON_PATH = MODELS_DIR / "resnet50_fc.json"
REGISTRY_PATH = MODELS_DIR / "registry.json"
EVAL_PATH = MODELS_DIR / "resnet50_onnx_eval.json"


def load_split(split: str) -> pd.DataFrame:
    """Rows of data/manifest.csv for `split`, with POSIX absolute filepaths."""
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["split"] == split].reset_index(drop=True)
    df["filepath"] = [
        str(PROJECT_ROOT / p.replace("\\", "/")) for p in df["filepath"]
    ]
    return df


def class_list(split: str = "test") -> list[str]:
    """Same rule as src/utils/dataset.py: sorted unique labels of the split."""
    return sorted(load_split(split)["label"].unique())


def load_classes() -> list[str]:
    return json.loads(CLASSES_PATH.read_text())


def build_torch_model():
    """ResNet50 with 80-class fc, weights from the best checkpoint, eval mode, CPU."""
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.resnet50 import build_resnet50

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = build_resnet50(num_classes=80, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
