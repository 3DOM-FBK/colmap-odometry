#!/usr/bin/env bash
# Exports the learned features to ONNX, once (needs Docker and internet):
#   xfeat.onnx                                  XFeat
#   aliked-n16_dense.onnx, aliked-n16_desc.onnx ALIKED (aliked-n16)
# The original PyTorch code and weights are fetched at pinned commits by
# Dockerfile.export; each export checks ONNX Runtime against PyTorch.
#
#   models/export_models.sh [reference_image]
#
# With a reference image (grayscale), the official PyTorch output for it is
# also written (*.reference.txt), to test the C++ implementation.
set -euo pipefail
cd "$(dirname "$0")"

docker build -t colmap-odometry-export -f Dockerfile.export .
REF=()
MOUNT=()
if [ $# -ge 1 ]; then
    IMG="$(realpath "$1")"
    MOUNT=(-v "$(dirname "$IMG")":/reference:ro)
    REF=(--reference "/reference/$(basename "$IMG")")
fi
RUN=(docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/models -w /models
     "${MOUNT[@]}" colmap-odometry-export)
"${RUN[@]}" python export_xfeat.py --out xfeat.onnx "${REF[@]}"
"${RUN[@]}" python export_aliked.py "${REF[@]}"
ls -la *.onnx
