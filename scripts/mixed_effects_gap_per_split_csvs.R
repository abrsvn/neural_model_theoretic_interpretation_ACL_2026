#!/usr/bin/env Rscript
#
# Mixed-effects per-split generalization-gap CSV generation
#
# Writes the per-split gap result files:
#   - results_gap_per_split_lrt_summary.csv
#   - results_gap_per_split_observation_counts.csv
#   - results_gap_per_split_<group>_<split>_arch.csv
#   - results_gap_per_split_<group>_<split>_entity.csv
#   - results_gap_per_split_<group>_<split>_arch_by_entity.csv
#   - results_gap_per_split_<group>_<split>_entity_by_arch.csv
#
# It intentionally does NOT emit:
#   - compact appendix TeX summaries
#   - combined-model gap CSVs
#   - diagnostic figures

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

options(width = 120, nwarnings = 10000, contrasts = c("contr.sum", "contr.poly"))
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 200000, pbkrtest.limit = 0)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript mixed_effects_gap_per_split_csvs.R <sentence_data.csv> <output_dir>\n")
  quit(status = 1)
}

data_file <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)


sig_stars <- function(p) {
  ifelse(is.na(p), "NA",
    ifelse(p < 0.001, "***",
      ifelse(p < 0.01, "**",
        ifelse(p < 0.05, "*", "ns"))))
}

format_p <- function(p) {
  ifelse(is.na(p), "NA",
    ifelse(p < 0.001, "<.001", sprintf("%.3f", p)))
}

write_warning_log <- function(output_dir, filename) {
  warning_path <- file.path(output_dir, filename)
  warns <- warnings()
  if (is.null(warns)) {
    writeLines("No warnings.", warning_path)
    cat(sprintf("Wrote warning log to %s (0 warnings)\n", warning_path))
    return()
  }

  warning_lines <- capture.output(warns)
  writeLines(warning_lines, warning_path)
  cat(sprintf("Wrote warning log to %s (%d warnings)\n",
              warning_path, length(warns)))
}

merge_ci <- function(summary_df, emm_contrasts) {
  ci <- as.data.frame(confint(emm_contrasts))
  summary_df$lower.CL <- ci$lower.CL
  summary_df$upper.CL <- ci$upper.CL
  summary_df
}

summarize_familywise_contrasts <- function(emm_contrasts, adjust) {
  if (adjust == "mvt") {
    set.seed(20260418)
  }
  as.data.frame(summary(
    emm_contrasts,
    infer = c(TRUE, TRUE),
    by = NULL,
    adjust = adjust
  ))
}

load_combined_re_selection <- function(output_dir) {
  path <- file.path(output_dir, "results_combined_re_selection.csv")
  if (!file.exists(path)) {
    stop(sprintf("Combined RE selection file does not exist: %s", path))
  }
  rows <- read.csv(path, stringsAsFactors = FALSE)
  if (!"selected_key" %in% names(rows)) {
    stop(sprintf("Combined RE selection file is missing selected_key: %s", path))
  }
  split(rows, rows$group)
}

build_per_split_re <- function(selection_row) {
  selected_key <- selection_row$selected_key[1]
  if (is.na(selected_key) || selected_key == "") {
    stop("Combined RE selection row is missing selected_key")
  }

  use_cov <- grepl("\\+cov$", selected_key)
  base_key <- sub("\\+cov$", "", selected_key)
  terms <- if (base_key == "baseline") character(0) else strsplit(base_key, "\\+")[[1]]
  terms <- setdiff(terms, "splt")

  if (use_cov) {
    sentence_terms <- intersect(terms, c("ent", "rec"))
    sentence_vars <- c(ent = "entity", rec = "is_recurrent")
    sentence_re <- if (length(sentence_terms) == 0) {
      "(1|sentence)"
    } else {
      paste0(
        "(1 + ",
        paste(unname(sentence_vars[sentence_terms]), collapse = " + "),
        " | sentence)"
      )
    }
    parts <- c(sentence_re)
    if ("arch" %in% terms) {
      parts <- c(parts, "(1|arch:sentence)")
    }
    if ("gap" %in% terms) {
      parts <- c(parts, "(1 + train_or_test | model_id)")
    } else {
      parts <- c(parts, "(1|model_id)")
    }
    return(paste(parts, collapse = " + "))
  }

  parts <- c("(1|sentence)", "(1|model_id)")
  if ("ent" %in% terms) {
    parts <- c(parts, "(0+entity|sentence)")
  }
  if ("rec" %in% terms) {
    parts <- c(parts, "(0+is_recurrent|sentence)")
  }
  if ("arch" %in% terms) {
    parts <- c(parts, "(1|arch:sentence)")
  }
  if ("gap" %in% terms) {
    parts <- c(parts, "(0+train_or_test|model_id)")
  }
  paste(parts, collapse = " + ")
}


cat("Loading data from", data_file, "\n")
data <- read.csv(data_file, stringsAsFactors = FALSE)

arch_levels <- unique(data$arch)
data$arch <- factor(data$arch, levels = arch_levels)
data$entity <- factor(data$entity, levels = c("noent", "ent"))
data$split <- factor(data$split)
data$sentence <- factor(data$sentence)
data$model_id <- factor(data$model_id)
data$train_or_test <- factor(data$train_or_test, levels = c("train", "test"))
data$is_recurrent <- factor(
  ifelse(data$arch %in% c("SRN", "GRU", "LSTM"), "recurrent", "attention"),
  levels = c("attention", "recurrent")
)

groups <- c("word_group", "sentence_group", "complex_event", "basic_event")
group_names <- c("Word", "Sentence", "Complex Event", "Basic Event")
names(group_names) <- groups
split_levels <- levels(data$split)


combined_re <- load_combined_re_selection(output_dir)


cat("\n", strrep("#", 70), "\n")
cat("  PER-SPLIT GENERALIZATION GAP (train vs test within each split)\n")
cat(strrep("#", 70), "\n")

obs_rows <- list()
lrt_rows <- list()

for (i in seq_along(groups)) {
  group <- groups[i]
  gname <- group_names[[i]]

  for (sp in split_levels) {
    gdf <- droplevels(data[data$group == group & data$split == sp, ])
    selection_row <- combined_re[[group]]
    if (is.null(selection_row) || nrow(selection_row) != 1) {
      stop(sprintf("Expected one combined RE selection row for %s", gname))
    }
    re <- build_per_split_re(selection_row)

    cat("\n", strrep("-", 50), "\n")
    cat(" ", gname, sp, "(n =", nrow(gdf),
        ", train =", sum(gdf$train_or_test == "train"),
        ", test =", sum(gdf$train_or_test == "test"), ")\n")
    cat("  Combined RE:", selection_row$selected_re[1], "\n")
    cat("  Per-split RE:", re, "\n")
    cat(strrep("-", 50), "\n")

    m_add <- lmer(as.formula(paste0(
      "advantage ~ arch + entity + train_or_test + ", re)),
      data = gdf, REML = TRUE)
    m_arch_tot <- lmer(as.formula(paste0(
      "advantage ~ arch * train_or_test + entity + ", re)),
      data = gdf, REML = TRUE)
    m_ent_tot <- lmer(as.formula(paste0(
      "advantage ~ arch + entity * train_or_test + ", re)),
      data = gdf, REML = TRUE)
    m_both <- lmer(as.formula(paste0(
      "advantage ~ arch * entity * train_or_test + ", re)),
      data = gdf, REML = TRUE)

    comp_arch <- anova(m_add, m_arch_tot)
    comp_ent <- anova(m_add, m_ent_tot)
    p_arch <- comp_arch[2, "Pr(>Chisq)"]
    p_ent <- comp_ent[2, "Pr(>Chisq)"]

    emm_arch <- emmeans(m_both, pairwise ~ arch | train_or_test, adjust = "tukey")
    arch_df <- merge_ci(as.data.frame(summary(emm_arch$contrasts)), emm_arch$contrasts)
    emm_ent <- emmeans(m_both, pairwise ~ entity | train_or_test, adjust = "none")
    ent_df <- summarize_familywise_contrasts(
      emm_ent$contrasts,
      adjust = "mvt"
    )
    emm_arch_ent <- emmeans(
      m_both,
      pairwise ~ arch | train_or_test * entity,
      adjust = "tukey"
    )
    arch_ent_df <- merge_ci(as.data.frame(summary(emm_arch_ent$contrasts)), emm_arch_ent$contrasts)
    emm_ent_arch <- emmeans(
      m_both,
      pairwise ~ entity | train_or_test * arch,
      adjust = "none"
    )
    ent_arch_df <- summarize_familywise_contrasts(
      emm_ent_arch$contrasts,
      adjust = "mvt"
    )

    key <- paste(group, sp, sep = "_")
    write.csv(arch_df, file.path(output_dir, sprintf("results_gap_per_split_%s_arch.csv", key)), row.names = FALSE)
    write.csv(ent_df, file.path(output_dir, sprintf("results_gap_per_split_%s_entity.csv", key)), row.names = FALSE)
    write.csv(arch_ent_df, file.path(output_dir, sprintf("results_gap_per_split_%s_arch_by_entity.csv", key)), row.names = FALSE)
    write.csv(ent_arch_df, file.path(output_dir, sprintf("results_gap_per_split_%s_entity_by_arch.csv", key)), row.names = FALSE)

    obs_rows[[length(obs_rows) + 1]] <- data.frame(
      group = group,
      group_name = gname,
      split = sp,
      n_obs = nrow(gdf),
      n_train = sum(gdf$train_or_test == "train"),
      n_test = sum(gdf$train_or_test == "test"),
      n_sent = nlevels(gdf$sentence),
      n_model = nlevels(gdf$model_id),
      stringsAsFactors = FALSE
    )
    lrt_rows[[length(lrt_rows) + 1]] <- data.frame(
      group = group,
      group_name = gname,
      split = sp,
      chisq_arch = comp_arch[2, "Chisq"],
      df_arch = comp_arch[2, "Df"],
      p_arch = p_arch,
      chisq_entity = comp_ent[2, "Chisq"],
      df_entity = comp_ent[2, "Df"],
      p_entity = p_ent,
      stringsAsFactors = FALSE
    )

    cat(sprintf("Saved per-split gap CSVs for %s %s\n", gname, sp))
  }
}

write.csv(do.call(rbind, obs_rows),
          file.path(output_dir, "results_gap_per_split_observation_counts.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, lrt_rows),
          file.path(output_dir, "results_gap_per_split_lrt_summary.csv"),
          row.names = FALSE)

write_warning_log(output_dir, "results_gap_per_split_warnings.txt")
cat("\nDone.\n")
