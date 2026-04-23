#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_checkpoint_evaluation.sh"
bash "${SCRIPT_DIR}/regenerate_tex_outputs.sh"
