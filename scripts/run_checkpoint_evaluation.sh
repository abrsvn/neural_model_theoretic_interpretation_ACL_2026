#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
CPU_CORES="${CPU_CORES:-$(getconf _NPROCESSORS_ONLN)}"
export OMP_NUM_THREADS="${CPU_CORES}"
export MKL_NUM_THREADS="${CPU_CORES}"
export OPENBLAS_NUM_THREADS="${CPU_CORES}"
export VECLIB_MAXIMUM_THREADS="${CPU_CORES}"
export NUMEXPR_NUM_THREADS="${CPU_CORES}"
export TORCH_NUM_THREADS="${CPU_CORES}"
export TORCH_NUM_INTEROP_THREADS="${CPU_CORES}"

ANALYSIS_ROOT="${ROOT_DIR}/analysis_per_model"
mkdir -p "${ANALYSIS_ROOT}"

echo "[eval] CPU cores detected: ${CPU_CORES}"
echo "[eval] CPU thread limits: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} TORCH=${TORCH_NUM_THREADS}"

python - <<'PY' "${ROOT_DIR}" "${ANALYSIS_ROOT}"
import json
import subprocess
import sys
from pathlib import Path

root_dir = Path(sys.argv[1])
analysis_root = Path(sys.argv[2])
checkpoints_path = root_dir / "metadata" / "checkpoints.json"
checkpoints_index = json.loads(checkpoints_path.read_text())

for file_entry in sorted(
    checkpoints_index["checkpoints"],
    key=lambda item: (
        item["experiment_id"],
        item["model_type"],
        item["entity_condition"],
        int(item["split"]),
        int(item["model_index"]),
    ),
):
    run_name = (
        f"{file_entry['model_type']}__{file_entry['entity_condition']}"
        f"__s{file_entry['split']}__m{file_entry['model_index']}"
    )
    output_dir = analysis_root / file_entry["experiment_id"] / run_name
    checkpoint_path = root_dir / file_entry["destination_path"]
    checkpoint_stem = checkpoint_path.stem
    expected_outputs = (
        output_dir / f"{checkpoint_stem}_rows_extended.csv",
        output_dir / f"{checkpoint_stem}_rows_canonical.csv",
        output_dir / f"{checkpoint_stem}_group_summary.csv",
        output_dir / f"{checkpoint_stem}_run_summary.csv",
    )

    if all(path.exists() for path in expected_outputs):
        print(f"[eval-skip] {file_entry['experiment_id']} {run_name}")
        print(f"Using existing analysis {expected_outputs[-1]}")
        continue

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {file_entry['experiment_id']} {run_name}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "evaluate",
            "--experiment-id",
            file_entry["experiment_id"],
            "--model-type",
            file_entry["model_type"],
            "--entity-condition",
            file_entry["entity_condition"],
            "--split",
            str(file_entry["split"]),
            "--model-index",
            str(file_entry["model_index"]),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
PY
