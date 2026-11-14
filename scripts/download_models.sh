#!/usr/bin/env bash
# Placeholder: fetch model artifacts into models/.
#
# The ONNX files (models/resnet50.onnx ~95 MB FP32, models/resnet50.int8.onnx ~24 MB)
# are gitignored. They will be hosted on a free public Hugging Face model repo
# (or a GitHub Release) in Phase 7 of docs/platform/PLAN.md, and this script will
# download them from there.
#
# Until then, regenerate them locally from the training checkpoint
# (outputs/checkpoints/resnet50_best.pt, also not in git):
#
#   python -m venv venv && venv/bin/pip install -r scripts/requirements-export.txt
#   venv/bin/python scripts/export_onnx.py --quantize
#   venv/bin/python scripts/evaluate_onnx.py      # accuracy + int8 agreement
#
# Small artifacts (registry.json, classes.json, resnet50_fc.npz/.json) are committed.
set -euo pipefail
echo "download_models.sh: hosting not set up yet (planned for Phase 7)." >&2
echo "Run scripts/export_onnx.py locally instead; see comments in this file." >&2
exit 1
