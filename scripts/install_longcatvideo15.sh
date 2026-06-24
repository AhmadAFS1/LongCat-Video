#!/usr/bin/env bash
set -euo pipefail

PREFIX="${LONGCAT_INSTALL_PREFIX:-$HOME/longcat-video-15}"
REPO_URL="${LONGCAT_REPO_URL:-https://github.com/AhmadAFS1/LongCat-Video.git}"
REPO_DIR="${LONGCAT_REPO_DIR:-$PREFIX/LongCat-Video}"
CONDA_DIR="${LONGCAT_CONDA_DIR:-$PREFIX/miniforge}"
ENV_PREFIX="${LONGCAT_ENV_PREFIX:-$PREFIX/env}"
HF_HOME="${HF_HOME:-$PREFIX/hf-cache}"
DOWNLOAD_MODELS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      REPO_DIR="${LONGCAT_REPO_DIR:-$PREFIX/LongCat-Video}"
      CONDA_DIR="${LONGCAT_CONDA_DIR:-$PREFIX/miniforge}"
      ENV_PREFIX="${LONGCAT_ENV_PREFIX:-$PREFIX/env}"
      HF_HOME="${HF_HOME:-$PREFIX/hf-cache}"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --skip-models)
      DOWNLOAD_MODELS=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON="$ENV_PREFIX/bin/python"
PIP="$PYTHON -m pip"

log() {
  printf '\n[%s] %s\n' "$(date +'%H:%M:%S')" "$*"
}

install_os_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing OS packages"
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y \
        bash ca-certificates curl wget git git-lfs ffmpeg libsndfile1 \
        build-essential
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        bash ca-certificates curl wget git git-lfs ffmpeg libsndfile1 \
        build-essential
    else
      echo "apt-get is available but sudo/root is not. Install git, git-lfs, ffmpeg, libsndfile1, and build-essential manually." >&2
    fi
  fi
}

install_miniforge() {
  if [[ -x "$CONDA_DIR/bin/conda" ]]; then
    log "Using existing Conda at $CONDA_DIR"
    return
  fi

  log "Installing Miniforge at $CONDA_DIR"
  mkdir -p "$PREFIX"
  local installer="/tmp/miniforge-$(date +%s).sh"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$installer"
  bash "$installer" -b -p "$CONDA_DIR"
  rm -f "$installer"
}

clone_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "Updating LongCat-Video repo at $REPO_DIR"
    git -C "$REPO_DIR" fetch --depth=1 origin main
    git -C "$REPO_DIR" checkout main
    git -C "$REPO_DIR" pull --ff-only origin main
  else
    log "Cloning LongCat-Video repo to $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --single-branch --branch main "$REPO_URL" "$REPO_DIR"
  fi
}

create_env() {
  if [[ ! -x "$PYTHON" ]]; then
    log "Creating Python 3.10 environment at $ENV_PREFIX"
    "$CONDA_DIR/bin/conda" create -y -p "$ENV_PREFIX" python=3.10 pip
  fi

  log "Upgrading packaging tools"
  $PIP install --upgrade pip setuptools wheel packaging ninja
}

install_python_packages() {
  log "Installing PyTorch 2.6.0 CUDA 12.4 wheels"
  $PIP install \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

  log "Installing LongCat base requirements without FlashAttention"
  grep -v '^flash-attn==' "$REPO_DIR/requirements.txt" > /tmp/longcat_requirements_no_flash.txt
  $PIP install -r /tmp/longcat_requirements_no_flash.txt

  log "Installing FlashAttention prebuilt wheel"
  $PIP install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

  log "Installing Avatar requirements, filtering non-PyPI/bad pins"
  grep -v -e '^libsndfile1==' -e '^tritonserverclient==' "$REPO_DIR/requirements_avatar.txt" > /tmp/longcat_requirements_avatar_filtered.txt
  $PIP install -r /tmp/longcat_requirements_avatar_filtered.txt
  $PIP install accelerate==0.34.2
}

apply_single_gpu_patch() {
  log "Applying single-GPU CPU-offload compatibility patch"
  "$PYTHON" - "$REPO_DIR" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])
pipeline = repo / "longcat_video" / "pipeline_longcat_video_avatar.py"
demo = repo / "run_demo_avatar_single_audio_to_video.py"

text = pipeline.read_text()
text = text.replace(
    "prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state",
    "encoder_device = next(self.text_encoder.parameters()).device\n"
    "        prompt_embeds = self.text_encoder(text_input_ids.to(encoder_device), mask.to(encoder_device)).last_hidden_state",
)
text = text.replace(
    "if self.text_encoder is not None:\n            self.text_encoder = self.text_encoder.to(device, non_blocking=True)",
    "if self.text_encoder is not None and os.environ.get(\"LONGCAT_TEXT_ENCODER_DEVICE\", \"\").lower() != \"cpu\":\n"
    "            self.text_encoder = self.text_encoder.to(device, non_blocking=True)",
)
text = text.replace(
    "# ---- Whisper encoder：mel → hidden states ----\n        enc_chunks = []\n        for i in range(0, audio_features.shape[-1], ENC_CHUNK):\n            chunk_hs = self.audio_encoder.encoder(\n                audio_features[:, :, i: i + ENC_CHUNK].to(device),",
    "# ---- Whisper encoder：mel → hidden states ----\n        encoder_device = next(self.audio_encoder.parameters()).device\n"
    "        enc_chunks = []\n        for i in range(0, audio_features.shape[-1], ENC_CHUNK):\n"
    "            chunk_hs = self.audio_encoder.encoder(\n                audio_features[:, :, i: i + ENC_CHUNK].to(encoder_device),",
)
pipeline.write_text(text)

text = demo.read_text()
text = text.replace(
    "audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)",
    "audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type)\n"
    "    if os.environ.get(\"LONGCAT_AUDIO_ENCODER_DEVICE\", \"\").lower() != \"cpu\":\n"
    "        audio_encoder = audio_encoder.to(local_rank)",
)
demo.write_text(text)
PY
}

download_models() {
  if [[ "$DOWNLOAD_MODELS" -eq 0 ]]; then
    log "Skipping model download"
    return
  fi

  log "Downloading selected LongCat model weights"
  mkdir -p "$HF_HOME" "$REPO_DIR/weights"
  export HF_HOME
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

  "$ENV_PREFIX/bin/hf" download meituan-longcat/LongCat-Video \
    --local-dir "$REPO_DIR/weights/LongCat-Video" \
    --include 'tokenizer/*' 'text_encoder/*' 'vae/*' 'scheduler/*' 'config.json' 'model_index.json'

  "$ENV_PREFIX/bin/hf" download meituan-longcat/LongCat-Video-Avatar-1.5 \
    --local-dir "$REPO_DIR/weights/LongCat-Video-Avatar-1.5" \
    --include \
      'base_model_int8/*' 'lora/*' 'scheduler/*' 'vocal_separator/*' \
      'config.json' 'model_index.json' \
      'whisper-large-v3/config.json' 'whisper-large-v3/generation_config.json' \
      'whisper-large-v3/preprocessor_config.json' 'whisper-large-v3/tokenizer_config.json' \
      'whisper-large-v3/tokenizer.json' 'whisper-large-v3/special_tokens_map.json' \
      'whisper-large-v3/added_tokens.json' 'whisper-large-v3/normalizer.json' \
      'whisper-large-v3/merges.txt' 'whisper-large-v3/vocab.json' \
      'whisper-large-v3/model.safetensors'

  find "$REPO_DIR/weights" -path '*/.cache' -type d -prune -exec rm -rf {} +
}

write_smoke_test_assets() {
  log "Creating short smoke-test input"
  mkdir -p "$REPO_DIR/assets/avatar/test"
  ffmpeg -y -hide_banner -loglevel error \
    -i "$REPO_DIR/assets/avatar/single/man.mp3" -t 4 -ar 16000 -ac 1 \
    "$REPO_DIR/assets/avatar/test/man_4s.wav"
  cp "$REPO_DIR/assets/avatar/single/man.png" "$REPO_DIR/assets/avatar/test/man.png"
  "$PYTHON" - "$REPO_DIR/assets/avatar/test/single_4s.json" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
data = {
    "prompt": "A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene.",
    "cond_image": "assets/avatar/test/man.png",
    "cond_audio": {"person1": "assets/avatar/test/man_4s.wav"},
}
path.write_text(json.dumps(data, indent=4) + "\n")
PY
}

verify_imports() {
  log "Verifying imports and CUDA visibility"
  "$PYTHON" - <<'PY'
import torch
import flash_attn
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch")
PY
}

main() {
  install_os_packages
  install_miniforge
  clone_repo
  create_env
  install_python_packages
  apply_single_gpu_patch
  download_models
  write_smoke_test_assets
  verify_imports

  log "LongCatVideo Avatar 1.5 install complete"
  cat <<EOF

Repo: $REPO_DIR
Python: $PYTHON

Smoke test command:
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LONGCAT_TEXT_ENCODER_DEVICE=cpu
export LONGCAT_AUDIO_ENCODER_DEVICE=cpu
"$ENV_PREFIX/bin/torchrun" --nproc_per_node=1 run_demo_avatar_single_audio_to_video.py \\
  --context_parallel_size=1 \\
  --checkpoint_dir=./weights/LongCat-Video-Avatar-1.5 \\
  --stage_1=ai2v \\
  --input_json=assets/avatar/test/single_4s.json \\
  --output_dir=./outputs_avatar_single_test \\
  --resolution=480p \\
  --num_segments=1 \\
  --use_distill \\
  --model_type avatar-v1.5 \\
  --use_int8
EOF
}

main "$@"
