# "The Learnability of Model-Theoretic Interpretation Functions in Artificial Neural Networks": Reproduction Code

Contains the checkpoints, systematicity evaluation sentences, target-weight files, and evaluation scripts for the ACL 2026 paper "The Learnability of Model-Theoretic Interpretation Functions in Artificial Neural Networks" (Brasoveanu & Dotlacil, ACL 2026).

Paper PDF: [Learnability_of_Model_Theoretic_Interpretation_Functions_in_ANNs_ACL_2026.pdf](./Learnability_of_Model_Theoretic_Interpretation_Functions_in_ANNs_ACL_2026.pdf)

It supports:

- inspecting the ACL checkpoints in `models/`
- reevaluating those checkpoints on the four paper systematicity groups
- rebuilding 150-dimensional and 300-dimensional supervision targets from formulas
- regenerating the main paper comparison plots, detailed appendix plots, generalization-gap plots, and the main paper tables
- regenerating the compact statistical appendix inputs derived from those four-test reevaluation outputs
- regenerating training-curve and LR-schedule plots from lightweight JSON trajectories
- regenerating sentence-level comprehension advantage distribution plots and tables

The `data/*.csv` files contain only the rows needed for the evaluation, i.e., the relevant subset of the sentences for the four systematicity tests:

- `Word`
- `Sentence`
- `Complex_Event`
- `Basic_Event`

The included rows satisfy the selection rule:

- `consistent == True`
- non-empty `competing_events`

The repository is focused on reevaluation and result regeneration. It does not include model training, microworld generation, grammar generation, or Pool MCMC sampling.

## Directory Structure

```text
data/
  train_set1.csv
  train_set2.csv
  test_set1.csv
  test_set2.csv
weights/
  competitive_150_props.npz
  competitive_150_entities.npz
models/
  exp_1_entity_vectors/
  exp_1_entity_vectors_attn_followup/
  exp_1_entity_vectors_GRU_followup/
metadata/
  checkpoint_seeds.json
  experiments.json
  checkpoints.json
src/
  cli.py
  checkpoints.py
  data/
    dataset.py
    targets.py
    vocabulary.py
  evaluation/
    batching.py
    metrics.py
    reporting.py
    systematicity.py
  models/
    attention.py
    recurrent.py
  plots/
    paper_detailed_plots.py
    paper_plots.py
    paper_table.py
  cross_model/
    diagnostics.py
    metadata.py
    plotting.py
    sentence_analysis.py
    sentence_data.py
training_trajectories/
  exp_1_entity_vectors/
  exp_1_entity_vectors_attn_followup/
  exp_1_entity_vectors_GRU_followup/
scripts/
  run_checkpoint_evaluation.sh
  regenerate_tex_outputs.sh
  regenerate_all_outputs.sh
```

## Install

Use Python 3.10 or newer.

```bash
cd <repository-root>
pip install -e .
```

Run the included CLI and shell scripts from this checkout. This code is not a self-contained wheel: the `data/`, `metadata/`, `weights/`, and `models/` assets stay in the directory tree and are not packaged as installed resources.

Python dependencies:

- `torch`
- `numpy`
- `matplotlib`
- `scipy`

The included statistical appendix scripts also require `Rscript` plus:

- `lme4`
- `lmerTest`
- `emmeans`

## Experiments

It includes three experiment families:

- `exp_1_entity_vectors`
  - models: `SIMPLE_RN`, `SIMPLE_LSTM`, `ABS_ATTN`, `ROPE_ATTN`
  - 2 splits
  - 5 seeds per configuration
  - `no_entity` and `with_entity`
  - 750 epochs in the full paper runs

- `exp_1_entity_vectors_attn_followup`
  - models: `ABS_ATTN`, `ROPE_ATTN`
  - hidden size 80
  - 2 splits
  - 5 seeds per configuration
  - `no_entity` and `with_entity`
  - 750 epochs in the full paper runs

- `exp_1_entity_vectors_GRU_followup`
  - models: `SIMPLE_GRU`
  - hidden size 90
  - 2 splits
  - 5 seeds per configuration
  - `no_entity` and `with_entity`
  - 750 epochs in the full paper runs

For ease of exposition, the paper folds `exp_1_entity_vectors_GRU_followup` into Experiment 1, resulting in a main five-architecture comparison, while the H80 attention
family remains a separate follow-up comparison.

The main metadata files are:

- `metadata/experiments.json`
- `metadata/checkpoint_seeds.json`
- `metadata/checkpoints.json`

## CLI

Use the module entry point from the checkout:

```bash
PYTHONPATH=src python -m cli --help
```

It exposes evaluation and figure/table regeneration commands. It also includes training-curve plotting, LR-schedule analysis, and advantage-distribution analysis for the included runs.

Start with checkpoint evaluation:

```bash
bash scripts/run_checkpoint_evaluation.sh
```

Then regenerate the figures, tables, and appendix outputs used by the TeX source:

```bash
bash scripts/regenerate_tex_outputs.sh
```

To run both steps in sequence:

```bash
bash scripts/regenerate_all_outputs.sh
```

The compact statistical appendix scripts are also included.

## Generated outputs

The included scripts write generated outputs under these root-level directories:

- `analysis_per_model/`
- `plots/`
- `tables/`
- `statistical_analysis/`

This checkout includes generated outputs in those directories. The shell
scripts can refresh or regenerate them from the included checkpoints, metadata,
and evaluation rows.

## Not Included

It does not include:

- microworld generation
- hard and soft constraints
- Z3 related infrastructure
- Pool MCMC sampling
- grammar generation and parsing
- competitive neural model training
- checkpoint import/migration workflows
- unrelated model families and analysis frameworks that are not needed for the experiments reported in the paper.
