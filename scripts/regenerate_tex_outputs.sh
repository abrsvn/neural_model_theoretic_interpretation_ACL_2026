#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-${ROOT_DIR}/analysis_per_model}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/models}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
FIGURE_ROOT="${FIGURE_ROOT:-${ROOT_DIR}/plots}"
TABLE_ROOT="${TABLE_ROOT:-${ROOT_DIR}/tables}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/statistical_analysis}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${ROOT_DIR}/training_trajectories}"
SENTENCE_CSV="${OUTPUT_ROOT}/sentence_data.csv"

cd "${ROOT_DIR}"

MAIN_ANALYSIS_DIR="${ANALYSIS_ROOT}/exp_1_entity_vectors"
ATTN_ANALYSIS_DIR="${ANALYSIS_ROOT}/exp_1_entity_vectors_attn_followup"
GRU_ANALYSIS_DIR="${ANALYSIS_ROOT}/exp_1_entity_vectors_GRU_followup"

MAIN_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/exp_1_entity_vectors"
ATTN_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/exp_1_entity_vectors_attn_followup"
GRU_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/exp_1_entity_vectors_GRU_followup"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "${label} does not exist: ${path}" >&2
    exit 1
  fi
}

require_nonempty_matches() {
  local path="$1"
  local pattern="$2"
  local label="$3"
  if ! find "${path}" -name "${pattern}" -print -quit | grep -q .; then
    echo "${label} not found under ${path}" >&2
    exit 1
  fi
}

require_dir "${MAIN_ANALYSIS_DIR}" "Main-experiment analysis directory"
require_dir "${ATTN_ANALYSIS_DIR}" "Attention follow-up analysis directory"
require_dir "${GRU_ANALYSIS_DIR}" "GRU follow-up analysis directory"
require_dir "${MAIN_CHECKPOINT_DIR}" "Main-experiment checkpoint directory"
require_dir "${ATTN_CHECKPOINT_DIR}" "Attention follow-up checkpoint directory"
require_dir "${GRU_CHECKPOINT_DIR}" "GRU follow-up checkpoint directory"
require_dir "${DATA_ROOT}" "Competing-events data directory"
require_dir "${TRAJECTORY_ROOT}" "Training-trajectory root"

require_nonempty_matches "${MAIN_ANALYSIS_DIR}" "*_run_summary.csv" "Main-experiment run summaries"
require_nonempty_matches "${ATTN_ANALYSIS_DIR}" "*_run_summary.csv" "Attention follow-up run summaries"
require_nonempty_matches "${GRU_ANALYSIS_DIR}" "*_run_summary.csv" "GRU follow-up run summaries"
require_nonempty_matches "${MAIN_ANALYSIS_DIR}" "*_rows_extended.csv" "Main-experiment detailed rows"
require_nonempty_matches "${ATTN_ANALYSIS_DIR}" "*_rows_extended.csv" "Attention follow-up detailed rows"
require_nonempty_matches "${GRU_ANALYSIS_DIR}" "*_rows_extended.csv" "GRU follow-up detailed rows"
require_nonempty_matches "${MAIN_CHECKPOINT_DIR}" "*_best_model.pt" "Main-experiment checkpoints"
require_nonempty_matches "${ATTN_CHECKPOINT_DIR}" "*_best_model.pt" "Attention follow-up checkpoints"
require_nonempty_matches "${GRU_CHECKPOINT_DIR}" "*_best_model.pt" "GRU follow-up checkpoints"

mkdir -p "${FIGURE_ROOT}" "${TABLE_ROOT}" "${OUTPUT_ROOT}"

mapfile -t MAIN_SUMMARIES < <(
  find "${MAIN_ANALYSIS_DIR}" -name "*_run_summary.csv" | sort
)
mapfile -t ATTN_SUMMARIES < <(
  find "${ATTN_ANALYSIS_DIR}" -name "*_run_summary.csv" | sort
)
mapfile -t GRU_SUMMARIES < <(
  find "${GRU_ANALYSIS_DIR}" -name "*_run_summary.csv" | sort
)

mapfile -t MAIN_DETAIL_CSVS < <(
  find "${MAIN_ANALYSIS_DIR}" -name "*_rows_extended.csv" | sort
)
mapfile -t ATTN_DETAIL_CSVS < <(
  find "${ATTN_ANALYSIS_DIR}" -name "*_rows_extended.csv" | sort
)
mapfile -t GRU_DETAIL_CSVS < <(
  find "${GRU_ANALYSIS_DIR}" -name "*_rows_extended.csv" | sort
)

mapfile -t MAIN_CHECKPOINTS < <(
  find "${MAIN_CHECKPOINT_DIR}" -name "*_best_model.pt" | sort
)
mapfile -t ATTN_CHECKPOINTS < <(
  find "${ATTN_CHECKPOINT_DIR}" -name "*_best_model.pt" | sort
)
mapfile -t GRU_CHECKPOINTS < <(
  find "${GRU_CHECKPOINT_DIR}" -name "*_best_model.pt" | sort
)

echo "[plot] main experiment entity-vector figure"
python -m cli plot-entity \
  --summary-csvs "${MAIN_SUMMARIES[@]}" "${GRU_SUMMARIES[@]}" \
  --output-path "${FIGURE_ROOT}/entity_vector_comparison_extended.png" \
  --omit-title

echo
echo "[plot] attention follow-up entity-vector figure"
python -m cli plot-entity \
  --summary-csvs "${ATTN_SUMMARIES[@]}" \
  --output-path "${FIGURE_ROOT}/ATTN_H80_entity_vector_comparison_extended.png" \
  --experiment-id exp_1_entity_vectors_attn_followup \
  --omit-title

echo
echo "[plot] GRU follow-up entity-vector figure"
python -m cli plot-entity \
  --summary-csvs "${GRU_SUMMARIES[@]}" \
  --output-path "${FIGURE_ROOT}/GRU_entity_vector_comparison_extended.png" \
  --experiment-id exp_1_entity_vectors_GRU_followup \
  --omit-title

echo
echo "[plot] main experiment detailed entity-vector figure"
python -m cli plot-entity-detailed \
  --detail-csvs "${MAIN_DETAIL_CSVS[@]}" \
  --output-path "${FIGURE_ROOT}/entity_vector_comparison_extended_detailed.png" \
  --experiment-id exp_1_entity_vectors

echo
echo "[plot] attention follow-up detailed entity-vector figure"
python -m cli plot-entity-detailed \
  --detail-csvs "${ATTN_DETAIL_CSVS[@]}" \
  --output-path "${FIGURE_ROOT}/ATTN_H80_entity_vector_comparison_extended_detailed.png" \
  --experiment-id exp_1_entity_vectors_attn_followup

echo
echo "[plot] GRU follow-up detailed entity-vector figure"
python -m cli plot-entity-detailed \
  --detail-csvs "${GRU_DETAIL_CSVS[@]}" \
  --output-path "${FIGURE_ROOT}/GRU_entity_vector_comparison_extended_detailed.png" \
  --experiment-id exp_1_entity_vectors_GRU_followup

echo
echo "[plot] main experiment generalization-gap figures"
python -m cli plot-gap \
  --summary-csvs "${MAIN_SUMMARIES[@]}" \
  --output-dir "${FIGURE_ROOT}/generalization_gap_OG_models" \
  --experiment-id exp_1_entity_vectors

echo
echo "[plot] attention follow-up generalization-gap figures"
python -m cli plot-gap \
  --summary-csvs "${ATTN_SUMMARIES[@]}" \
  --output-dir "${FIGURE_ROOT}/generalization_gap_ATTN_H80_models" \
  --experiment-id exp_1_entity_vectors_attn_followup

echo
echo "[plot] GRU follow-up generalization-gap figures"
python -m cli plot-gap \
  --summary-csvs "${GRU_SUMMARIES[@]}" \
  --output-dir "${FIGURE_ROOT}/generalization_gap_GRU_models" \
  --experiment-id exp_1_entity_vectors_GRU_followup

echo
echo "[table] regenerate main table"
python -m cli paper-table \
  --summary-csvs "${MAIN_SUMMARIES[@]}" "${GRU_SUMMARIES[@]}" \
  --output-path "${TABLE_ROOT}/stat_descriptive.tex"

echo
echo "[build] cross-model sentence_data.csv"
python -m cli sentence-data \
  --analysis-dirs "${MAIN_ANALYSIS_DIR}" "${ATTN_ANALYSIS_DIR}" "${GRU_ANALYSIS_DIR}" \
  --output-path "${SENTENCE_CSV}"

echo
echo "[table] modifier-complexity breakdown"
python "${SCRIPT_DIR}/summarize_complexity_breakdown.py" \
  --sentence-csv "${SENTENCE_CSV}" \
  --output-path "${TABLE_ROOT}/stat_complexity_breakdown_paper.tex"

echo
echo "[table] Word/Sentence modifier disaggregation"
python "${SCRIPT_DIR}/summarize_word_sentence_disaggregation.py" \
  --sentence-csv "${SENTENCE_CSV}" \
  --output-path "${TABLE_ROOT}/stat_disagg_word_sentence_paper.tex"

echo
echo "[plot] competing-events appendix outputs"
python -m cli competing-events \
  --data-dir "${DATA_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"

echo
echo "[plot] distribution plots and tables"
python -m cli distribution \
  --sentence-csv "${SENTENCE_CSV}" \
  --output-dir "${OUTPUT_ROOT}"

echo
echo "[plot] descriptive tables"
python -m cli descriptive \
  --sentence-csv "${SENTENCE_CSV}" \
  --output-dir "${OUTPUT_ROOT}" \
  --paper-sentence-output-path "${TABLE_ROOT}/stat_descriptive_paper_sentence.tex"

echo
echo "[table] mixed-effects appendix outputs"
Rscript "${SCRIPT_DIR}/mixed_effects.R" "${SENTENCE_CSV}" "${OUTPUT_ROOT}"
python "${SCRIPT_DIR}/summarize_mixed_effects_tables.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/compact_mixed_effects_summaries"

echo
echo "[table] generalization-gap result CSVs"
Rscript "${SCRIPT_DIR}/mixed_effects_gap_csvs.R" "${SENTENCE_CSV}" "${OUTPUT_ROOT}"
echo "Warning log: ${OUTPUT_ROOT}/results_gap_combined_warnings.txt"

echo
echo "[table] per-split generalization-gap result CSVs"
Rscript "${SCRIPT_DIR}/mixed_effects_gap_per_split_csvs.R" "${SENTENCE_CSV}" "${OUTPUT_ROOT}"
echo "Warning log: ${OUTPUT_ROOT}/results_gap_per_split_warnings.txt"
python "${SCRIPT_DIR}/summarize_generalization_gap.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/compact_gap_summaries"

echo
echo "[table] sentence-difficulty and architecture-gap diagnostics"
Rscript "${SCRIPT_DIR}/mixed_effects_sentence_diagnostics.R" "${SENTENCE_CSV}" "${OUTPUT_ROOT}"
echo "Warning log: ${OUTPUT_ROOT}/results_sentence_diagnostics_warnings.txt"
python "${SCRIPT_DIR}/summarize_sentence_arch_gap.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/compact_sentence_diagnostics_summaries"
Rscript "${SCRIPT_DIR}/summarize_test_sentence_difficulty_types.R" \
  "${OUTPUT_ROOT}" \
  "${OUTPUT_ROOT}/compact_sentence_difficulty_summaries" \
  3

echo
echo "[plot] main experiment training-curve figures"
python -m cli plot-training \
  --checkpoint-paths "${MAIN_CHECKPOINTS[@]}" \
  --output-dir "${FIGURE_ROOT}/training_trajectories_OG_models" \
  --experiment-id exp_1_entity_vectors \
  --trajectory-root "${TRAJECTORY_ROOT}"

echo
echo "[plot] attention follow-up training-curve figures"
python -m cli plot-training \
  --checkpoint-paths "${ATTN_CHECKPOINTS[@]}" \
  --output-dir "${FIGURE_ROOT}/training_trajectories_ATTN_H80_models" \
  --experiment-id exp_1_entity_vectors_attn_followup \
  --trajectory-root "${TRAJECTORY_ROOT}"

echo
echo "[plot] GRU follow-up training-curve figures"
python -m cli plot-training \
  --checkpoint-paths "${GRU_CHECKPOINTS[@]}" \
  --output-dir "${FIGURE_ROOT}/training_trajectories_GRU_models" \
  --experiment-id exp_1_entity_vectors_GRU_followup \
  --trajectory-root "${TRAJECTORY_ROOT}"

echo
echo "[plot] LR schedules and best-epoch appendix outputs"
python -m cli lr-schedules \
  --trajectory-root "${TRAJECTORY_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
