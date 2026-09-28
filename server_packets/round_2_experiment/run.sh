#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="/srv/nfs/home/njnu_lhf/aic-repechage/project"
DATA_ROOT="/srv/nfs/data/aic-repechage/official/repechage"
WORK_ROOT="/srv/nfs/home/njnu_lhf/aic-repechage"
PREFERRED_ENV="$WORK_ROOT/env"
FALLBACK_ENV="$WORK_ROOT/env_rgb_baseline"
EXTERNAL_ROOT="$WORK_ROOT/external"
GDINO_ROOT="$EXTERNAL_ROOT/GroundingDINO"
GDINO_COMMIT="856dde20aee659246248e20734ef9ba5214f5e44"
WEIGHTS_ROOT="$WORK_ROOT/weights"
WEIGHT_PATH="$WEIGHTS_ROOT/groundingdino_swinb_cogcoor.pth"
TRANSFERRED_WEIGHT_PATH="$WORK_ROOT/groundingdino_swinb_cogcoor.pth"
EXPECTED_WEIGHT_SHA="46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7"
WEIGHT_GITHUB_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
WEIGHT_HF_URL="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
OUTPUT_BASE="$WORK_ROOT/outputs"
HF_HOME="$WORK_ROOT/cache/huggingface"
PIP_CACHE_DIR="$WORK_ROOT/cache/pip"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PAYLOAD_DIR="$SCRIPT_DIR/payload"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESUME_OUTPUT=""

if [[ "${1:-}" == "--resume-output" ]]; then
    RESUME_OUTPUT="${2:-}"
    [[ -n "$RESUME_OUTPUT" ]] || { echo "--resume-output requires a path" >&2; exit 2; }
elif [[ $# -gt 0 ]]; then
    echo "Unsupported arguments: $*" >&2
    exit 2
fi

mkdir -p "$OUTPUT_BASE"
RESULT_DIR="$OUTPUT_BASE/round_2_handoff_$TIMESTAMP"
mkdir -p "$RESULT_DIR"
STATUS_FILE="$RESULT_DIR/STATUS.txt"
COMMANDS_LOG="$RESULT_DIR/commands.log"
STDOUT_LOG="$RESULT_DIR/stdout.log"
STDERR_LOG="$RESULT_DIR/stderr.log"
CURRENT_STAGE="initialization"
EXPERIMENT_OUTPUT="${RESUME_OUTPUT:-$OUTPUT_BASE/rgb_baseline_v1_$TIMESTAMP}"
ARCHIVE_PATH="$RESULT_DIR.tar.gz"

printf 'INCOMPLETE\n' >"$STATUS_FILE"
: >"$COMMANDS_LOG"
: >"$STDOUT_LOG"
: >"$STDERR_LOG"

exec > >(tee -a "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)

log() { printf '%s\n' "$*"; }
stage() { CURRENT_STAGE="$1"; printf '\n===== STAGE: %s =====\n' "$CURRENT_STAGE"; }
run() {
    printf '[%s] ' "$CURRENT_STAGE" >>"$COMMANDS_LOG"
    printf '%q ' "$@" >>"$COMMANDS_LOG"
    printf '\n' >>"$COMMANDS_LOG"
    "$@"
}

collect_results() {
    local destination="$RESULT_DIR/experiment"
    mkdir -p "$destination"
    if [[ -d "$EXPERIMENT_OUTPUT" ]]; then
        find "$EXPERIMENT_OUTPUT" -maxdepth 1 -type f \
            \( -name '*.json' -o -name '*.jsonl' \) -exec cp -f {} "$destination/" \; || true
        for config_dir in "$EXPERIMENT_OUTPUT"/box*; do
            [[ -d "$config_dir" ]] || continue
            mkdir -p "$destination/$(basename "$config_dir")"
            find "$config_dir" -maxdepth 1 -type f \
                \( -name '*.json' -o -name '*.jsonl' \) \
                -exec cp -f {} "$destination/$(basename "$config_dir")/" \; || true
        done
        if [[ -d "$EXPERIMENT_OUTPUT/visualizations" ]]; then
            cp -a "$EXPERIMENT_OUTPUT/visualizations" "$destination/" || true
        fi
    fi
    printf '%s\n' "$CURRENT_STAGE" >"$RESULT_DIR/last_stage.txt"
    printf '%s\n' "$EXPERIMENT_OUTPUT" >"$RESULT_DIR/experiment_output_path.txt"
}

finalize() {
    local exit_status=$?
    trap - EXIT
    set +e
    if [[ $exit_status -eq 0 ]]; then
        printf 'PASS\n' >"$STATUS_FILE"
    else
        printf 'FAIL\n' >"$STATUS_FILE"
    fi
    collect_results
    if command -v sha256sum >/dev/null 2>&1; then
        (
            cd "$RESULT_DIR" || exit 1
            find . -type f ! -name checksums.txt -print0 | sort -z | xargs -0 sha256sum
        ) >"$RESULT_DIR/checksums.txt" 2>/dev/null
    fi
    if command -v tar >/dev/null 2>&1; then
        tar -czf "$ARCHIVE_PATH" -C "$OUTPUT_BASE" "$(basename "$RESULT_DIR")" 2>/dev/null
    fi
    printf 'ROUND2_STATUS=%s\n' "$(tr -d '\r\n' <"$STATUS_FILE")"
    printf 'ROUND2_RESULT_DIR=%s\n' "$RESULT_DIR"
    if [[ -f "$ARCHIVE_PATH" ]]; then
        printf 'ROUND2_ARCHIVE=%s\n' "$ARCHIVE_PATH"
    else
        printf 'ROUND2_ARCHIVE=NOT_CREATED\n'
    fi
    exit "$exit_status"
}
trap finalize EXIT

python_version_compatible() {
    "$1" - <<'PY'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 12) else 1)
PY
}

write_environment_report() {
    local python_bin="$1"
    local output="$2"
    "$python_bin" - <<'PY' >"$output"
import importlib.util
import json
import os
import sys

result = {"python": sys.version, "executable": sys.executable, "modules": {}}
for name in ("numpy", "cv2", "PIL", "yaml", "torch", "torchvision", "transformers", "timm", "groundingdino"):
    result["modules"][name] = importlib.util.find_spec(name) is not None
try:
    import torch
    result["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "visible_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception as exc:
    result["torch_error"] = f"{type(exc).__name__}: {exc}"
result["cuda_visible_devices_set"] = "CUDA_VISIBLE_DEVICES" in os.environ
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
}

download_file() {
    local destination="$1"
    shift
    local partial="$destination.part"
    local url
    for url in "$@"; do
        log "Trying official download: $url"
        if command -v curl >/dev/null 2>&1; then
            if curl -L --fail --retry 3 --continue-at - -o "$partial" "$url"; then
                mv -f "$partial" "$destination"
                return 0
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -c -O "$partial" "$url"; then
                mv -f "$partial" "$destination"
                return 0
            fi
        else
            log "Neither curl nor wget is available"
            return 1
        fi
    done
    return 1
}

stage "preflight"
[[ "$(cd "$PROJECT_ROOT" && pwd -P)" == "$PROJECT_ROOT" ]] || {
    echo "Project root mismatch" >&2
    exit 1
}
cd "$PROJECT_ROOT"
[[ -r "$DATA_ROOT/queries/queries.json" ]] || {
    echo "Official queries JSON is not readable" >&2
    exit 1
}
[[ -d "$PAYLOAD_DIR" ]] || { echo "Execution package payload is missing" >&2; exit 1; }
run df -h "$PROJECT_ROOT" "$DATA_ROOT" "$WORK_ROOT"
run nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    SELECTED_GPU="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t',' -k2 -nr | head -n 1 | cut -d',' -f1 | tr -d ' '
    )"
    [[ -n "$SELECTED_GPU" ]] || { echo "Unable to select a GPU" >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
    log "Selected physical GPU with most free memory: $SELECTED_GPU"
else
    log "Respecting existing CUDA_VISIBLE_DEVICES assignment"
fi
mkdir -p "$EXTERNAL_ROOT" "$WEIGHTS_ROOT" "$OUTPUT_BASE" "$HF_HOME" "$PIP_CACHE_DIR"
export HF_HOME PIP_CACHE_DIR TOKENIZERS_PARALLELISM=false

stage "select_python_environment"
ENV_PREFIX=""
if [[ -x "$PREFERRED_ENV/bin/python" ]] && python_version_compatible "$PREFERRED_ENV/bin/python"; then
    ENV_PREFIX="$PREFERRED_ENV"
elif [[ -x "$FALLBACK_ENV/bin/python" ]] && python_version_compatible "$FALLBACK_ENV/bin/python"; then
    ENV_PREFIX="$FALLBACK_ENV"
else
    run conda create -y -p "$FALLBACK_ENV" python=3.10 pip
    ENV_PREFIX="$FALLBACK_ENV"
fi
PYTHON_BIN="$ENV_PREFIX/bin/python"
PIP_BIN=("$PYTHON_BIN" -m pip)
write_environment_report "$PYTHON_BIN" "$RESULT_DIR/environment_before.json"
run "$PYTHON_BIN" --version

stage "install_python_dependencies"
TORCH_OK=false
if "$PYTHON_BIN" - <<'PY'
try:
    import torch, torchvision
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if torch.__version__.startswith("2.3.") and torchvision.__version__.startswith("0.18.") else 1)
PY
then
    TORCH_OK=true
fi
if [[ "$TORCH_OK" != true ]]; then
    run "${PIP_BIN[@]}" install \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121
fi
run "${PIP_BIN[@]}" install \
    numpy==1.26.4 pillow==10.4.0 opencv-python-headless==4.10.0.84 \
    'pyyaml>=6,<7' 'pytest>=8,<10' 'transformers>=4.39,<4.58' \
    'timm>=0.9,<2' 'addict>=2.4,<3' 'yapf>=0.40,<1' 'supervision>=0.22.0' \
    'pycocotools>=2.0.7,<3' 'termcolor>=2,<4'
run "${PIP_BIN[@]}" install --no-deps -e "$PROJECT_ROOT"

stage "install_project_payload"
OLD_BASELINE_SHA="3f045524e2e48490f47a1060d2483128802b77ab98b0cb8ff4ebf828051fcc75"
NEW_BASELINE_SHA="cb0250a88258f98f86ac0e547e96f41258bfb6f36db20191e36191b48742e5d8"
CURRENT_BASELINE_SHA="$(sha256sum "$PROJECT_ROOT/baseline/inference.py" | awk '{print $1}')"
if [[ "$CURRENT_BASELINE_SHA" != "$OLD_BASELINE_SHA" && "$CURRENT_BASELINE_SHA" != "$NEW_BASELINE_SHA" ]]; then
    echo "baseline/inference.py has unknown changes; refusing to overwrite" >&2
    exit 1
fi
install -m 0644 "$PAYLOAD_DIR/baseline/inference.py" "$PROJECT_ROOT/baseline/inference.py"
install -m 0644 "$PAYLOAD_DIR/scripts/run_repechage_rgb_experiment.py" \
    "$PROJECT_ROOT/scripts/run_repechage_rgb_experiment.py"
install -m 0644 "$PAYLOAD_DIR/tests/test_repechage_experiment.py" \
    "$PROJECT_ROOT/tests/test_repechage_experiment.py"
printf 'installed\n' >"$RESULT_DIR/project_payload_status.txt"
run "${PIP_BIN[@]}" install --no-deps -e "$PROJECT_ROOT"

stage "prepare_groundingdino_source"
if [[ -d "$GDINO_ROOT/.git" ]]; then
    run git -C "$GDINO_ROOT" fetch --all --tags
    run git -C "$GDINO_ROOT" checkout --detach "$GDINO_COMMIT"
elif [[ -f "$GDINO_ROOT/.codex_pinned_commit" ]]; then
    log "Reusing extracted GroundingDINO source at $GDINO_ROOT"
elif [[ -f "$SCRIPT_DIR/assets/GroundingDINO-$GDINO_COMMIT.tar.gz" ]]; then
    [[ ! -e "$GDINO_ROOT" ]] || { echo "$GDINO_ROOT exists but is not a Git repository" >&2; exit 1; }
    mkdir -p "$GDINO_ROOT"
    run tar -xzf "$SCRIPT_DIR/assets/GroundingDINO-$GDINO_COMMIT.tar.gz" \
        --strip-components=1 -C "$GDINO_ROOT"
    printf '%s\n' "$GDINO_COMMIT" >"$GDINO_ROOT/.codex_pinned_commit"
else
    [[ ! -e "$GDINO_ROOT" ]] || { echo "$GDINO_ROOT exists but is not usable" >&2; exit 1; }
    run git clone https://github.com/IDEA-Research/GroundingDINO.git "$GDINO_ROOT"
    run git -C "$GDINO_ROOT" checkout --detach "$GDINO_COMMIT"
fi
if [[ -d "$GDINO_ROOT/.git" ]]; then
    ACTUAL_GDINO_COMMIT="$(git -C "$GDINO_ROOT" rev-parse HEAD)"
else
ACTUAL_GDINO_COMMIT="$(cat "$GDINO_ROOT/.codex_pinned_commit")"
fi
[[ "$ACTUAL_GDINO_COMMIT" == "$GDINO_COMMIT" ]] || {
    echo "GroundingDINO commit mismatch" >&2
    exit 1
}
printf '%s\n' "$ACTUAL_GDINO_COMMIT" >"$RESULT_DIR/groundingdino_commit.txt"

OFFLINE_BERT_PATCH="$SCRIPT_DIR/patches/get_tokenlizer.py"
[[ -f "$OFFLINE_BERT_PATCH" ]] || {
    echo "Offline BERT loader patch is missing: $OFFLINE_BERT_PATCH" >&2
    exit 1
}
run install -m 0644 "$OFFLINE_BERT_PATCH" \
    "$GDINO_ROOT/groundingdino/util/get_tokenlizer.py"
sha256sum "$OFFLINE_BERT_PATCH" >"$RESULT_DIR/offline_bert_patch_sha256.txt"

NVCC_PATH="$(command -v nvcc || true)"
if [[ -z "$NVCC_PATH" ]]; then
    NVCC_PATH="$(find /usr/local -maxdepth 4 -type f -path '*/cuda*/bin/nvcc' -print 2>/dev/null | sort | tail -n 1 || true)"
fi
[[ -n "$NVCC_PATH" ]] || {
    echo "nvcc was not found; GroundingDINO CUDA extension cannot be built" >&2
    exit 1
}
export CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd -P)"
log "CUDA_HOME=$CUDA_HOME"
run "${PIP_BIN[@]}" install --no-deps --no-build-isolation -e "$GDINO_ROOT"

stage "prepare_swinb_weight"
if [[ -f "$SCRIPT_DIR/assets/groundingdino_swinb_cogcoor.pth" ]]; then
    run cp -f "$SCRIPT_DIR/assets/groundingdino_swinb_cogcoor.pth" "$WEIGHT_PATH"
elif [[ -f "$TRANSFERRED_WEIGHT_PATH" ]]; then
    log "Using separately transferred official weight: $TRANSFERRED_WEIGHT_PATH"
    run cp -f "$TRANSFERRED_WEIGHT_PATH" "$WEIGHT_PATH"
fi
if [[ ! -f "$WEIGHT_PATH" ]]; then
    download_file "$WEIGHT_PATH" "$WEIGHT_GITHUB_URL" "$WEIGHT_HF_URL" || {
        echo "Unable to download official Swin-B weight" >&2
        exit 1
    }
fi
WEIGHT_SIZE="$(stat -c '%s' "$WEIGHT_PATH")"
[[ "$WEIGHT_SIZE" -gt 500000000 ]] || {
    echo "Weight file is unexpectedly small: $WEIGHT_SIZE bytes" >&2
    exit 1
}
ACTUAL_WEIGHT_SHA="$(sha256sum "$WEIGHT_PATH" | awk '{print $1}')"
[[ "$ACTUAL_WEIGHT_SHA" == "$EXPECTED_WEIGHT_SHA" ]] || {
    echo "Weight SHA-256 mismatch: expected $EXPECTED_WEIGHT_SHA, got $ACTUAL_WEIGHT_SHA" >&2
    exit 1
}
run sha256sum "$WEIGHT_PATH"
sha256sum "$WEIGHT_PATH" >"$RESULT_DIR/weight_sha256.txt"

stage "verify_environment_and_tests"
write_environment_report "$PYTHON_BIN" "$RESULT_DIR/environment_after.json"
run "$PYTHON_BIN" -c 'import cv2, pycocotools.mask, torch, torchvision, groundingdino, groundingdino._C; from groundingdino.util.inference import load_model; print(torch.__version__, torchvision.__version__, cv2.__version__)'
run "$PYTHON_BIN" -m pytest -q -p no:cacheprovider
if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    run "$PYTHON_BIN" scripts/check_repo_safety.py
else
    log "Repository safety script skipped on server because project has no .git metadata; local package safety check is required."
fi

stage "run_model_and_threshold_experiment"
MODEL_CONFIG="$GDINO_ROOT/groundingdino/config/GroundingDINO_SwinB_cfg.py"
[[ -f "$MODEL_CONFIG" ]] || { echo "Swin-B config missing: $MODEL_CONFIG" >&2; exit 1; }
EXPERIMENT_ARGS=(
    "$PYTHON_BIN" scripts/run_repechage_rgb_experiment.py
    --data-root "$DATA_ROOT"
    --model-config "$MODEL_CONFIG"
    --weights "$WEIGHT_PATH"
    --groundingdino-root "$GDINO_ROOT"
    --output-root "$EXPERIMENT_OUTPUT"
    --max-samples 100
    --visualization-count 20
)
if [[ -n "$RESUME_OUTPUT" ]]; then
    EXPERIMENT_ARGS+=(--resume)
fi
run "${EXPERIMENT_ARGS[@]}"

stage "complete"
[[ -f "$EXPERIMENT_OUTPUT/experiment_summary.json" ]] || {
    echo "Experiment summary was not created" >&2
    exit 1
}
log "Experiment completed successfully: $EXPERIMENT_OUTPUT"
