#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_longcat_avatar15.sh --image IMAGE --audio AUDIO --output-dir OUTPUT_DIR [--prompt PROMPT]

Environment overrides:
  LONGCAT_TORCHRUN                 Path to torchrun.
  LONGCAT_CHECKPOINT_DIR           Avatar 1.5 checkpoint dir.
  LONGCAT_CONTEXT_PARALLEL_SIZE    Defaults to 1.
  LONGCAT_TEXT_ENCODER_DEVICE      Defaults to cpu.
  LONGCAT_AUDIO_ENCODER_DEVICE     Defaults to cpu.
EOF
}

IMAGE=""
AUDIO=""
OUTPUT_DIR=""
PROMPT="A person is speaking naturally, with realistic facial motion and synchronized mouth movement."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --audio)
      AUDIO="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$IMAGE" || -z "$AUDIO" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_DIR="${LONGCAT_CHECKPOINT_DIR:-$REPO_DIR/weights/LongCat-Video-Avatar-1.5}"
CONTEXT_PARALLEL_SIZE="${LONGCAT_CONTEXT_PARALLEL_SIZE:-1}"

if [[ -n "${LONGCAT_TORCHRUN:-}" ]]; then
  TORCHRUN="$LONGCAT_TORCHRUN"
elif [[ -x "$REPO_DIR/../env/bin/torchrun" ]]; then
  TORCHRUN="$REPO_DIR/../env/bin/torchrun"
elif [[ -x "/venv/longcat-video/bin/torchrun" ]]; then
  TORCHRUN="/venv/longcat-video/bin/torchrun"
else
  TORCHRUN="$(command -v torchrun)"
fi

mkdir -p "$OUTPUT_DIR"
INPUT_JSON="$OUTPUT_DIR/input.json"

python3 - "$IMAGE" "$AUDIO" "$PROMPT" "$INPUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

image, audio, prompt, out = sys.argv[1:]
data = {
    "prompt": prompt,
    "cond_image": str(Path(image).expanduser().resolve()),
    "cond_audio": {"person1": str(Path(audio).expanduser().resolve())},
}
Path(out).write_text(json.dumps(data, indent=4) + "\n")
PY

cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LONGCAT_TEXT_ENCODER_DEVICE="${LONGCAT_TEXT_ENCODER_DEVICE:-cpu}"
export LONGCAT_AUDIO_ENCODER_DEVICE="${LONGCAT_AUDIO_ENCODER_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$TORCHRUN" --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \
  --context_parallel_size="$CONTEXT_PARALLEL_SIZE" \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --stage_1=ai2v \
  --input_json="$INPUT_JSON" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=480p \
  --num_segments=1 \
  --use_distill \
  --model_type avatar-v1.5 \
  --use_int8
