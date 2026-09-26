#!/usr/bin/env bash

# Read-only audit for the AIC repechage server. The only writes are audit
# artifacts under OUTPUT_ROOT (default: /tmp). No network, install, Git write,
# dataset modification, or model-weight copy is performed.

set -u
set -o pipefail

EXPECTED_PROJECT_ROOT="/srv/nfs/home/njnu_lhf/aic-repechage/project"
DATA_ROOT="/srv/nfs/data/aic-repechage/official/repechage"
OUTPUT_ROOT="${TMPDIR:-/tmp}"

if [[ "${1:-}" == "--output-root" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "ERROR: --output-root requires a directory" >&2
        exit 2
    fi
    OUTPUT_ROOT="$2"
elif [[ $# -gt 0 ]]; then
    echo "ERROR: unsupported arguments: $*" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="$OUTPUT_ROOT/aic_repechage_round_1_audit_$TIMESTAMP"
ARCHIVE_PATH="$RESULT_DIR.tar.gz"

mkdir -p -- "$RESULT_DIR" || {
    echo "ERROR: cannot create audit result directory: $RESULT_DIR" >&2
    exit 1
}

COMMANDS_LOG="$RESULT_DIR/commands.log"
STDOUT_LOG="$RESULT_DIR/stdout.log"
STDERR_LOG="$RESULT_DIR/stderr.log"
REPORT="$RESULT_DIR/report.txt"
STATUS_FILE="$RESULT_DIR/STATUS.txt"
SUMMARY_JSON="$RESULT_DIR/summary.json"

: >"$COMMANDS_LOG"
: >"$STDOUT_LOG"
: >"$STDERR_LOG"
: >"$REPORT"
printf 'INCOMPLETE\n' >"$STATUS_FILE"

log() {
    printf '%s\n' "$*" | tee -a "$STDOUT_LOG"
}

section() {
    printf '\n===== %s =====\n' "$1" | tee -a "$STDOUT_LOG" "$REPORT"
}

run_capture() {
    local label="$1"
    shift
    printf '[%s] ' "$label" >>"$COMMANDS_LOG"
    printf '%q ' "$@" >>"$COMMANDS_LOG"
    printf '\n' >>"$COMMANDS_LOG"
    printf '\n--- %s ---\n' "$label" | tee -a "$STDOUT_LOG" "$REPORT"
    "$@" > >(tee -a "$STDOUT_LOG" "$REPORT") 2> >(tee -a "$STDERR_LOG" >&2)
    local status=$?
    printf '[exit=%d] %s\n' "$status" "$label" | tee -a "$STDOUT_LOG" "$REPORT"
    return 0
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

on_exit() {
    local status=$?
    if [[ $status -ne 0 ]]; then
        printf 'FAIL\n' >"$STATUS_FILE"
        printf 'run.sh exited with status %d\n' "$status" >>"$STDERR_LOG"
    fi
}
trap on_exit EXIT

log "AIC repechage round-1 read-only audit"
log "RESULT_DIR=$RESULT_DIR"
log "PROJECT_ROOT=$PROJECT_ROOT"
log "DATA_ROOT=$DATA_ROOT"

section "Host and clock"
run_capture "hostname" hostname
run_capture "kernel" uname -a
run_capture "identity" id
run_capture "UTC clock" date -u '+%Y-%m-%dT%H:%M:%SZ'
run_capture "working directory" pwd -P

section "Project identity"
printf 'expected_project_root=%s\nactual_project_root=%s\n' \
    "$EXPECTED_PROJECT_ROOT" "$PROJECT_ROOT" | tee -a "$REPORT" "$STDOUT_LOG"
if [[ "$PROJECT_ROOT" == "$EXPECTED_PROJECT_ROOT" ]]; then
    PROJECT_PATH_MATCH=true
else
    PROJECT_PATH_MATCH=false
fi

if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    IS_GIT_REPOSITORY=true
    run_capture "Git top level" git -C "$PROJECT_ROOT" rev-parse --show-toplevel
    run_capture "Git remotes" git -C "$PROJECT_ROOT" remote -v
    run_capture "Git branch" git -C "$PROJECT_ROOT" branch --show-current
    run_capture "Git HEAD" git -C "$PROJECT_ROOT" rev-parse HEAD
    git -C "$PROJECT_ROOT" status --short --branch >"$RESULT_DIR/git_status.txt" 2>>"$STDERR_LOG" || true
    run_capture "Git status" cat "$RESULT_DIR/git_status.txt"
    run_capture "Recent commits" git -C "$PROJECT_ROOT" log -5 --oneline --decorate
else
    IS_GIT_REPOSITORY=false
    printf 'Not a Git work tree: %s\n' "$PROJECT_ROOT" | tee -a "$STDERR_LOG" "$REPORT"
    printf 'NOT_A_GIT_REPOSITORY\n' >"$RESULT_DIR/git_status.txt"
fi

section "Filesystem and data root"
if [[ -d "$DATA_ROOT" && -r "$DATA_ROOT" ]]; then
    DATA_ROOT_READABLE=true
else
    DATA_ROOT_READABLE=false
fi
printf 'data_root_exists=%s\ndata_root_readable=%s\n' \
    "$([[ -d "$DATA_ROOT" ]] && echo true || echo false)" \
    "$DATA_ROOT_READABLE" | tee -a "$REPORT" "$STDOUT_LOG"
run_capture "Disk space" df -h "$PROJECT_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT"
run_capture "Project parent permissions" ls -ld "$PROJECT_ROOT" "$(dirname -- "$PROJECT_ROOT")"
if [[ -e "$DATA_ROOT" ]]; then
    run_capture "Data root permissions" ls -ld "$DATA_ROOT"
fi

JSON_LIST="$RESULT_DIR/json_candidates.txt"
: >"$JSON_LIST"
if [[ "$DATA_ROOT_READABLE" == true ]]; then
    find "$DATA_ROOT" -maxdepth 5 -type f \
        \( -iname '*.json' -o -iname '*.jsonl' \) -print 2>>"$STDERR_LOG" \
        | sort | head -n 30 >"$JSON_LIST"
fi
run_capture "Candidate JSON files (maximum 30)" cat "$JSON_LIST"

PYTHON_BIN=""
if command_exists python3; then
    PYTHON_BIN="$(command -v python3)"
elif command_exists python; then
    PYTHON_BIN="$(command -v python)"
fi

if [[ -n "$PYTHON_BIN" ]]; then
    mapfile -t JSON_FILES <"$JSON_LIST"
    run_capture "Dataset JSON structure and first image metadata" \
        "$PYTHON_BIN" "$SCRIPT_DIR/scripts/inspect_dataset.py" \
        --data-root "$DATA_ROOT" \
        --output "$RESULT_DIR/dataset_summary.json" \
        "${JSON_FILES[@]}"
else
    printf '{"error":"python executable not found"}\n' >"$RESULT_DIR/dataset_summary.json"
    printf 'Python executable not found; dataset structure inspection skipped.\n' \
        | tee -a "$STDERR_LOG" "$REPORT"
fi

section "GPU, Python, and ML runtime"
if command_exists nvidia-smi; then
    NVIDIA_SMI_AVAILABLE=true
    run_capture "NVIDIA GPU summary" nvidia-smi \
        --query-gpu=index,name,driver_version,memory.total,memory.free \
        --format=csv,noheader
    run_capture "NVIDIA SMI full header" nvidia-smi
else
    NVIDIA_SMI_AVAILABLE=false
    printf 'nvidia-smi not found\n' | tee -a "$STDERR_LOG" "$REPORT"
fi

if [[ -n "$PYTHON_BIN" ]]; then
    run_capture "Python version" "$PYTHON_BIN" --version
    run_capture "Python executable" "$PYTHON_BIN" -c \
        'import sys; print(sys.executable); print(sys.version)'
    "$PYTHON_BIN" - <<'PY' >"$RESULT_DIR/python_runtime.txt" 2>>"$STDERR_LOG"
import importlib.util
import json
import sys

result = {
    "executable": sys.executable,
    "version": sys.version,
    "modules": {},
}
for name in ("numpy", "cv2", "PIL", "yaml", "torch", "torchvision", "groundingdino"):
    result["modules"][name] = {"available": importlib.util.find_spec(name) is not None}

try:
    import torch

    result["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception as exc:
    result["torch_error"] = f"{type(exc).__name__}: {exc}"

try:
    import torchvision

    result["torchvision_version"] = torchvision.__version__
except Exception as exc:
    result["torchvision_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result, ensure_ascii=False, indent=2))
PY
    run_capture "Python ML runtime" cat "$RESULT_DIR/python_runtime.txt"
else
    printf 'Python executable not found\n' >"$RESULT_DIR/python_runtime.txt"
fi

if command_exists conda; then
    run_capture "Conda executable" command -v conda
    run_capture "Conda environments" conda info --envs
else
    printf 'conda not found on current PATH\n' | tee -a "$REPORT" "$STDOUT_LOG"
fi

section "GroundingDINO source and configuration candidates"
GDINO_CANDIDATES="$RESULT_DIR/groundingdino_candidates.txt"
: >"$GDINO_CANDIDATES"
SEARCH_ROOTS=("$PROJECT_ROOT" "$(dirname -- "$PROJECT_ROOT")" "$HOME")
if [[ -d /srv/nfs/home/njnu_lhf ]]; then
    SEARCH_ROOTS+=("/srv/nfs/home/njnu_lhf")
fi

for root in "${SEARCH_ROOTS[@]}"; do
    [[ -d "$root" && -r "$root" ]] || continue
    find "$root" -maxdepth 6 \
        \( -type d -iname 'GroundingDINO' \
        -o -type f -iname '*GroundingDINO*SwinB*.py' \
        -o -type f -iname '*grounding*dino*swin*b*.py' \) \
        -print 2>>"$STDERR_LOG" || true
done | awk '!seen[$0]++' | head -n 50 >"$GDINO_CANDIDATES"
run_capture "GroundingDINO candidates" cat "$GDINO_CANDIDATES"

while IFS= read -r candidate; do
    [[ -d "$candidate/.git" ]] || continue
    printf '\n[%s]\n' "$candidate" | tee -a "$REPORT" "$STDOUT_LOG"
    run_capture "GroundingDINO candidate HEAD" git -C "$candidate" rev-parse HEAD
    run_capture "GroundingDINO candidate remote" git -C "$candidate" remote -v
done <"$GDINO_CANDIDATES"

section "Swin-B weight candidates"
WEIGHT_CANDIDATES="$RESULT_DIR/weight_candidates.txt"
: >"$WEIGHT_CANDIDATES"
WEIGHT_SEARCH_ROOTS=("$PROJECT_ROOT" "$(dirname -- "$PROJECT_ROOT")")
if [[ -d /srv/nfs/home/njnu_lhf ]]; then
    WEIGHT_SEARCH_ROOTS+=("/srv/nfs/home/njnu_lhf")
fi
if [[ -d /srv/nfs/data/aic-repechage ]]; then
    WEIGHT_SEARCH_ROOTS+=("/srv/nfs/data/aic-repechage")
fi

for root in "${WEIGHT_SEARCH_ROOTS[@]}"; do
    [[ -d "$root" && -r "$root" ]] || continue
    find "$root" -maxdepth 7 -type f \
        \( -iname 'groundingdino_swinb_cogcoor.pth' \
        -o -iname '*grounding*dino*swin*b*.pth' \
        -o -iname '*groundingdino*swinb*.pth' \) \
        -print 2>>"$STDERR_LOG" || true
done | awk '!seen[$0]++' | head -n 10 >"$WEIGHT_CANDIDATES"
run_capture "Weight candidates (maximum 10)" cat "$WEIGHT_CANDIDATES"

if command_exists sha256sum; then
    while IFS= read -r weight; do
        [[ -f "$weight" ]] || continue
        run_capture "Weight metadata" stat -- "$weight"
        run_capture "Weight SHA-256" sha256sum -- "$weight"
    done <"$WEIGHT_CANDIDATES"
elif command_exists shasum; then
    while IFS= read -r weight; do
        [[ -f "$weight" ]] || continue
        run_capture "Weight metadata" stat -- "$weight"
        run_capture "Weight SHA-256" shasum -a 256 -- "$weight"
    done <"$WEIGHT_CANDIDATES"
else
    printf 'No SHA-256 command found; weight hashes unavailable.\n' \
        | tee -a "$STDERR_LOG" "$REPORT"
fi

section "Repository manifests"
for manifest in README.md pyproject.toml requirements.txt environment.yml configs/baseline.yaml; do
    path="$PROJECT_ROOT/$manifest"
    if [[ -f "$path" ]]; then
        run_capture "Manifest: $manifest" sed -n '1,220p' "$path"
    fi
done

WEIGHT_COUNT="$(grep -cve '^$' "$WEIGHT_CANDIDATES" 2>/dev/null || true)"
GDINO_COUNT="$(grep -cve '^$' "$GDINO_CANDIDATES" 2>/dev/null || true)"
JSON_COUNT="$(grep -cve '^$' "$JSON_LIST" 2>/dev/null || true)"
PYTHON_AVAILABLE=false
[[ -n "$PYTHON_BIN" ]] && PYTHON_AVAILABLE=true

if [[ -n "$PYTHON_BIN" ]]; then
    PROJECT_ROOT_VALUE="$PROJECT_ROOT" \
    EXPECTED_PROJECT_ROOT_VALUE="$EXPECTED_PROJECT_ROOT" \
    DATA_ROOT_VALUE="$DATA_ROOT" \
    PROJECT_PATH_MATCH_VALUE="$PROJECT_PATH_MATCH" \
    IS_GIT_REPOSITORY_VALUE="$IS_GIT_REPOSITORY" \
    DATA_ROOT_READABLE_VALUE="$DATA_ROOT_READABLE" \
    NVIDIA_SMI_AVAILABLE_VALUE="$NVIDIA_SMI_AVAILABLE" \
    PYTHON_AVAILABLE_VALUE="$PYTHON_AVAILABLE" \
    JSON_COUNT_VALUE="$JSON_COUNT" \
    GDINO_COUNT_VALUE="$GDINO_COUNT" \
    WEIGHT_COUNT_VALUE="$WEIGHT_COUNT" \
    "$PYTHON_BIN" - <<'PY' >"$SUMMARY_JSON"
import json
import os

def truth(name: str) -> bool:
    return os.environ[name].lower() == "true"

summary = {
    "audit_version": 1,
    "project_root": os.environ["PROJECT_ROOT_VALUE"],
    "expected_project_root": os.environ["EXPECTED_PROJECT_ROOT_VALUE"],
    "project_path_match": truth("PROJECT_PATH_MATCH_VALUE"),
    "is_git_repository": truth("IS_GIT_REPOSITORY_VALUE"),
    "data_root": os.environ["DATA_ROOT_VALUE"],
    "data_root_readable": truth("DATA_ROOT_READABLE_VALUE"),
    "nvidia_smi_available": truth("NVIDIA_SMI_AVAILABLE_VALUE"),
    "python_available": truth("PYTHON_AVAILABLE_VALUE"),
    "json_candidate_count": int(os.environ["JSON_COUNT_VALUE"] or 0),
    "groundingdino_candidate_count": int(os.environ["GDINO_COUNT_VALUE"] or 0),
    "weight_candidate_count": int(os.environ["WEIGHT_COUNT_VALUE"] or 0),
    "readiness": "AUDITED",
    "note": "AUDITED means the read-only audit completed; it does not mean every runtime asset is ready.",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
else
    cat >"$SUMMARY_JSON" <<EOF
{"audit_version":1,"readiness":"INCOMPLETE","error":"python executable not found"}
EOF
fi

printf 'PASS\n' >"$STATUS_FILE"

section "Audit summary"
run_capture "Summary JSON" cat "$SUMMARY_JSON"
log "AUDIT_STATUS=$(tr -d '\r\n' <"$STATUS_FILE")"
log "AUDIT_RESULT_DIR=$RESULT_DIR"
log "AUDIT_ARCHIVE_EXPECTED=$ARCHIVE_PATH"

CHECKSUMS="$RESULT_DIR/checksums.txt"
if command_exists sha256sum; then
    (
        cd "$RESULT_DIR" || exit 1
        find . -maxdepth 1 -type f ! -name checksums.txt -print0 \
            | sort -z | xargs -0 sha256sum
    ) >"$CHECKSUMS" 2>>"$STDERR_LOG" || true
elif command_exists shasum; then
    (
        cd "$RESULT_DIR" || exit 1
        find . -maxdepth 1 -type f ! -name checksums.txt -print0 \
            | sort -z | xargs -0 shasum -a 256
    ) >"$CHECKSUMS" 2>>"$STDERR_LOG" || true
else
    printf 'SHA-256 utility unavailable\n' >"$CHECKSUMS"
fi

if command_exists tar; then
    tar -czf "$ARCHIVE_PATH" -C "$OUTPUT_ROOT" "$(basename -- "$RESULT_DIR")" \
        2>>"$STDERR_LOG" || true
fi

trap - EXIT
printf 'AUDIT_STATUS=%s\n' "$(tr -d '\r\n' <"$STATUS_FILE")"
printf 'AUDIT_RESULT_DIR=%s\n' "$RESULT_DIR"
if [[ -f "$ARCHIVE_PATH" ]]; then
    printf 'AUDIT_ARCHIVE=%s\n' "$ARCHIVE_PATH"
else
    printf 'AUDIT_ARCHIVE=NOT_CREATED\n'
fi
