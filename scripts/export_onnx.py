"""
Phase 0: export the trained ResNet50 to ONNX + portable side artifacts.

Outputs (in models/):
  resnet50.onnx        FP32, inputs `input` (N,3,224,224); outputs
                       `logits` (N,80) and `features` (N,2048,7,7) from layer4
  resnet50.int8.onnx   optional int8 dynamic-quantized variant (--quantize)
  resnet50_fc.npz      fc weight (80,2048) / bias (80,) float32 for CAM
  resnet50_fc.json     same, as JSON (flat row-major arrays) for the browser
  classes.json         80 class names in training index order
  registry.json        model registry

Usage (from repo root):
    venv/bin/python scripts/export_onnx.py [--quantize]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import (  # noqa: E402
    CLASSES_PATH, EVAL_PATH, FC_JSON_PATH, FC_NPZ_PATH, MODELS_DIR, ONNX_INT8_PATH,
    ONNX_PATH, PROJECT_ROOT, REGISTRY_PATH, build_torch_model, class_list,
)
from preprocessing import DEFAULT_MEAN, DEFAULT_SIZE, DEFAULT_STD  # noqa: E402

OPSET = 17


class ResNet50WithFeatures(nn.Module):
    """Returns (logits, layer4 feature map). Logits are computed exactly as in
    torchvision's ResNet.forward, so they are bit-identical to model(x)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

    def forward(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        feats = x
        logits = m.fc(torch.flatten(m.avgpool(x), 1))
        return logits, feats


def rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def export_fp32(model: nn.Module) -> None:
    wrapper = ResNet50WithFeatures(model).eval()
    dummy = torch.randn(1, 3, DEFAULT_SIZE, DEFAULT_SIZE)
    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print(f"wrote {rel(ONNX_PATH)} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def export_int8() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = MODELS_DIR / "resnet50.pre.onnx"
    quant_pre_process(str(ONNX_PATH), str(pre), skip_symbolic_shape=True)
    quantize_dynamic(
        str(pre), str(ONNX_INT8_PATH), weight_type=QuantType.QUInt8, per_channel=False
    )
    pre.unlink(missing_ok=True)
    print(f"wrote {rel(ONNX_INT8_PATH)} ({ONNX_INT8_PATH.stat().st_size / 1e6:.1f} MB)")


def write_fc(model: nn.Module) -> None:
    w = model.fc.weight.detach().cpu().numpy().astype(np.float32)
    b = model.fc.bias.detach().cpu().numpy().astype(np.float32)
    assert w.shape == (80, 2048) and b.shape == (80,)
    np.savez(FC_NPZ_PATH, weight=w, bias=b)
    FC_JSON_PATH.write_text(json.dumps({
        "shape": list(w.shape),
        "layout": "row-major: weight[c*2048 + k] = W[c, k]",
        # float32 -> shortest repr that round-trips through float32
        "weight": [float(np.format_float_positional(v, unique=True)) for v in w.ravel()],
        "bias": [float(v) for v in b],
    }, separators=(",", ":")))
    print(f"wrote {rel(FC_NPZ_PATH)} ({FC_NPZ_PATH.stat().st_size / 1e6:.2f} MB), "
          f"{rel(FC_JSON_PATH)} ({FC_JSON_PATH.stat().st_size / 1e6:.2f} MB)")


def write_registry(int8_usable: bool | None) -> None:
    variants = {"fp32": {"onnx": rel(ONNX_PATH), "usable": True}}
    if ONNX_INT8_PATH.exists():
        variants["int8"] = {
            "onnx": rel(ONNX_INT8_PATH),
            "usable": bool(int8_usable),
            "note": "int8 dynamic quantization (uint8 per-tensor weights; ConvInteger/MatMulInteger ops); usable only if "
                    "test-set top-1 agreement with fp32 >= 99% (see eval file)",
        }
    registry = {
        "default": "resnet50",
        "models": [{
            "id": "resnet50",
            "display_name": "ResNet50",
            "architecture": "torchvision resnet50, fc = Linear(2048, 80)",
            "onnx": rel(ONNX_PATH),
            "variants": variants,
            "inputs": {"name": "input", "layout": "NCHW", "dtype": "float32"},
            "outputs": {"logits": "N x 80", "features": "N x 2048 x 7 x 7 (layer4)"},
            "classes": rel(CLASSES_PATH),
            "num_classes": 80,
            "fc": {"npz": rel(FC_NPZ_PATH), "json": rel(FC_JSON_PATH)},
            "input_size": [DEFAULT_SIZE, DEFAULT_SIZE],
            "resize": "bilinear, direct to 224x224 (no crop, aspect not preserved)",
            "scale": "pixel / 255",
            "mean": list(DEFAULT_MEAN),
            "std": list(DEFAULT_STD),
            "metrics": {
                "results": "outputs/results/resnet50_results.json",
                "evaluation": "outputs/results/resnet50_evaluation.json",
                "onnx_eval": rel(EVAL_PATH),
            },
            "cam": {
                "layer": "layer4",
                "method": "CAM == Grad-CAM for GAP->Linear head: "
                          "heatmap_c = ReLU(sum_k W_fc[c,k] * features[k]); "
                          "upsample 7x7 -> 224x224, normalize to [0,1]",
            },
        }],
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"wrote {rel(REGISTRY_PATH)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true", help="also write int8 variant")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    model, ckpt = build_torch_model()
    print(f"checkpoint epoch={ckpt['epoch']} best_val_acc={ckpt['best_val_acc']:.4f}")

    classes = class_list("test")
    assert classes == class_list("train") == class_list("val") and len(classes) == 80
    CLASSES_PATH.write_text(json.dumps(classes, indent=2) + "\n")
    print(f"wrote {rel(CLASSES_PATH)} ({len(classes)} classes)")

    export_fp32(model)
    write_fc(model)
    if args.quantize:
        export_int8()

    # int8 usability is decided by scripts/evaluate_onnx.py, which rewrites the registry.
    usable = None
    if EVAL_PATH.exists():
        usable = json.loads(EVAL_PATH.read_text()).get("int8", {}).get("usable")
    write_registry(usable)


if __name__ == "__main__":
    main()
