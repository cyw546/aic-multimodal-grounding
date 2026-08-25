from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_SIZE = 5 * 1024 * 1024
FORBIDDEN_ROOTS = {
    "data", "checkpoints", "outputs", "submissions", "logs", "cache",
    "wandb", "runs",
}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine",
    ".pem", ".key", ".zip",
}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(password|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
]


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    files = candidate_files()
    problems: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if relative.parts and relative.parts[0].lower() in FORBIDDEN_ROOTS:
            problems.append(f"forbidden repository root: {relative}")
        if path.name == ".env" or path.name.startswith(".env."):
            if path.name != ".env.example":
                problems.append(f"forbidden environment file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file type: {relative}")
        if path.stat().st_size > MAX_SIZE:
            problems.append(f"file exceeds 5 MB: {relative}")
        if (
            path.stat().st_size <= 1024 * 1024
            and relative != Path("scripts/check_repo_safety.py")
        ):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"possible secret: {relative}")
                    break

    if problems:
        print("REPOSITORY SAFETY CHECK: FAIL")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1

    print("REPOSITORY SAFETY CHECK: PASS")
    print(f"Checked {len(files)} candidate files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
