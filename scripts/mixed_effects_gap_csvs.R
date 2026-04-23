#!/usr/bin/env Rscript
#
# Mixed-effects generalization-gap CSV generation
#
# Generates the combined train+test generalization-gap results:
#   - results_combined_re_selection.csv
#   - results_gap_<group>_anova.csv
#   - results_gap_<group>_arch_pairwise.csv
#   - results_gap_<group>_entity.csv
#   - results_gap_<group>_arch_pairwise_by_entity.csv
#   - results_gap_<group>_entity_by_arch.csv
#
# It intentionally does NOT emit:
#   - compact appendix TeX summaries
#   - per-split gap CSVs
#   - diagnostic figures

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

options(width = 120, nwarnings = 10000, contrasts = c("contr.sum", "contr.poly"))
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 200000, pbkrtest.limit = 0)


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript mixed_effects_gap_csvs.R <sentence_data.csv> <output_dir>\n")
  quit(status = 1)
}

data_file <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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

backward_eliminate_re <- function(gdf, fe_str) {
  comp_formulas <- c(
    ent = "(0+entity|sentence)",
    rec = "(0+is_recurrent|sentence)",
    splt = "(0+split|sentence)",
    arch = "(1|arch:sentence)",
    gap = "(0+train_or_test|model_id)"
  )

  build_re <- function(active) {
    parts <- c("(1|sentence)", "(1|model_id)")
    for (nm in active) parts <- c(parts, comp_formulas[nm])
    paste(parts, collapse = " + ")
  }

  try_fit <- function(re) {
    tryCatch(
      lmer(as.formula(paste0(fe_str, " + ", re)), data = gdf, REML = TRUE),
      error = function(e) NULL
    )
  }

  all_terms <- names(comp_formulas)
  all_combos <- list()
  for (k in seq(length(all_terms), 0)) {
    if (k == 0) {
      all_combos <- c(all_combos, list(character(0)))
    } else {
      all_combos <- c(all_combos, combn(all_terms, k, simplify = FALSE))
    }
  }

  fits <- list()
  info <- list()

  for (combo in all_combos) {
    key <- if (length(combo) == 0) "baseline" else paste(sort(combo), collapse = "+")
    re <- build_re(combo)
    cat(sprintf("  %-15s", key))
    fit <- try_fit(re)
    fits[[key]] <- fit
    if (!is.null(fit)) {
      sing <- isSingular(fit)
      cat(sprintf(" AIC=%.1f%s\n", AIC(fit), if (sing) " [SINGULAR]" else ""))
      info[[key]] <- list(
        terms = combo,
        re = re,
        converged = TRUE,
        singular = sing,
        aic = AIC(fit)
      )
    } else {
      cat(" FAILED\n")
      info[[key]] <- list(
        terms = combo,
        re = re,
        converged = FALSE,
        singular = NA,
        aic = NA
      )
    }
  }

  base_fit <- fits[["baseline"]]
  lrt_vs_base <- list()
  if (!is.null(base_fit)) {
    for (key in names(fits)) {
      if (key == "baseline" || is.null(fits[[key]])) next
      comp <- tryCatch(anova(base_fit, fits[[key]]), error = function(e) NULL)
      if (!is.null(comp)) {
        lrt_vs_base[[key]] <- list(
          chisq = comp[2, "Chisq"],
          df = comp[2, "Df"],
          p = comp[2, "Pr(>Chisq)"]
        )
      }
    }
  }

  valid_keys <- names(info)[sapply(info, function(x) x$converged && !x$singular)]
  if (length(valid_keys) == 0 ||
      (length(valid_keys) == 1 && valid_keys[1] == "baseline")) {
    cat("  No non-singular alternatives, using baseline\n")
    return(list(
      selected_re = build_re(character(0)),
      selected_key = "baseline",
      info = info,
      lrt_vs_base = lrt_vs_base,
      reason = "no non-singular alternatives"
    ))
  }

  n_terms <- sapply(info[valid_keys], function(x) length(x$terms))
  max_n <- max(n_terms)
  richest <- valid_keys[n_terms == max_n]
  richest_aics <- sapply(info[richest], function(x) x$aic)
  start_key <- richest[which.min(richest_aics)]

  current_key <- start_key
  current_terms <- info[[current_key]]$terms
  cat(sprintf("\n  Backward elimination from: %s\n", current_key))

  repeat {
    if (length(current_terms) == 0) break

    drop_candidates <- list()
    for (term in current_terms) {
      simpler <- setdiff(current_terms, term)
      skey <- if (length(simpler) == 0) "baseline" else paste(sort(simpler), collapse = "+")
      if (is.null(fits[[skey]]) || !info[[skey]]$converged) next

      comp <- tryCatch(anova(fits[[skey]], fits[[current_key]]), error = function(e) NULL)
      if (is.null(comp)) next

      pv <- comp[2, "Pr(>Chisq)"]
      cat(sprintf("    drop %-5s: Chisq=%.2f df=%d p=%s %s%s\n",
                  term, comp[2, "Chisq"], comp[2, "Df"],
                  format_p(pv), sig_stars(pv),
                  if (info[[skey]]$singular) " [SINGULAR]" else ""))
      drop_candidates[[term]] <- list(
        term = term,
        key = skey,
        p = pv,
        singular = info[[skey]]$singular
      )
    }

    if (length(drop_candidates) == 0) break

    ok <- Filter(function(d) d$p >= 0.05 && !d$singular, drop_candidates)
    if (length(ok) == 0) {
      cat("    All removals significant or singular -> stop\n")
      break
    }

    best <- ok[[which.max(sapply(ok, function(d) d$p))]]
    cat(sprintf("    Dropping '%s' (p=%.3f) -> %s\n", best$term, best$p, best$key))
    current_key <- best$key
    current_terms <- info[[current_key]]$terms
  }

  final_re <- info[[current_key]]$re
  final_key <- current_key
  sentence_slope_terms <- intersect(current_terms, c("ent", "rec", "splt"))

  if (length(sentence_slope_terms) >= 1 || "gap" %in% current_terms) {
    slope_map <- c(ent = "entity", rec = "is_recurrent", splt = "split")
    cov_parts <- c()
    if (length(sentence_slope_terms) >= 1) {
      slope_vars <- unname(slope_map[sentence_slope_terms])
      cov_parts <- c(
        cov_parts,
        paste0("(1 + ", paste(slope_vars, collapse = " + "), " | sentence)")
      )
    } else {
      cov_parts <- c(cov_parts, "(1|sentence)")
    }
    if ("arch" %in% current_terms) cov_parts <- c(cov_parts, "(1|arch:sentence)")
    if ("gap" %in% current_terms) {
      cov_parts <- c(cov_parts, "(1 + train_or_test | model_id)")
    } else {
      cov_parts <- c(cov_parts, "(1|model_id)")
    }
    cov_re <- paste(cov_parts, collapse = " + ")

    cat(sprintf("\n  Covariance upgrade: %s\n", cov_re))
    cov_fit <- try_fit(cov_re)
    if (!is.null(cov_fit) && !isSingular(cov_fit)) {
      comp <- tryCatch(anova(fits[[current_key]], cov_fit), error = function(e) NULL)
      if (!is.null(comp)) {
        pv <- comp[2, "Pr(>Chisq)"]
        cat(sprintf("    LRT: Chisq=%.2f df=%d p=%s %s\n",
                    comp[2, "Chisq"], comp[2, "Df"], format_p(pv), sig_stars(pv)))
        if (pv < 0.05) {
          cat("    Accepted\n")
          final_re <- cov_re
          final_key <- paste0(current_key, "+cov")
          fits[[final_key]] <- cov_fit
          info[[final_key]] <- list(
            terms = current_terms,
            re = cov_re,
            converged = TRUE,
            singular = FALSE,
            aic = AIC(cov_fit)
          )
          if (!is.null(base_fit)) {
            comp_b <- tryCatch(anova(base_fit, cov_fit), error = function(e) NULL)
            if (!is.null(comp_b)) {
              lrt_vs_base[[final_key]] <- list(
                chisq = comp_b[2, "Chisq"],
                df = comp_b[2, "Df"],
                p = comp_b[2, "Pr(>Chisq)"]
              )
            }
          }
        } else {
          cat("    Not significant, keeping diagonal\n")
        }
      }
    } else {
      if (!is.null(cov_fit)) cat("    Singular\n") else cat("    Failed\n")
    }
  }

  cat(sprintf("\n  SELECTED: %s\n", final_re))

  list(
    selected_re = final_re,
    selected_key = final_key,
    info = info,
    lrt_vs_base = lrt_vs_base,
    reason = sprintf("backward elimination from %s", start_key)
  )
}


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

cat("Loading data from", data_file, "\n")
data <- read.csv(data_file, stringsAsFactors = FALSE)

arch_levels <- unique(data$arch)
data$arch <- factor(data$arch, levels = arch_levels)
data$entity <- factor(data$entity, levels = c("noent", "ent"))
data$split <- factor(data$split)
data$sentence <- factor(data$sentence)
data$model_id <- factor(data$model_id)
data$train_or_test <- factor(data$train_or_test, levels = c("train", "test"))

# Group architectures into recurrent vs attention.
data$is_recurrent <- factor(
  ifelse(data$arch %in% c("SRN", "GRU", "LSTM"), "recurrent", "attention"),
  levels = c("attention", "recurrent")
)

cat("Loaded:", nrow(data), "rows,",
    nlevels(data$arch), "architectures,",
    nlevels(data$model_id), "models\n")
cat("Train/test split:", table(data$train_or_test), "\n\n")

groups <- c("word_group", "sentence_group", "complex_event", "basic_event")
group_names <- c("Word", "Sentence", "Complex Event", "Basic Event")
names(group_names) <- groups


# ---------------------------------------------------------------------------
# Random-effects selection on the combined data
# ---------------------------------------------------------------------------

re_selected <- list(combined = list())
re_comparisons <- list(combined = list())
fe_gap <- "advantage ~ arch * entity * train_or_test + split"

cat("\n", strrep("#", 70), "\n")
cat("  Random effects selection (combined data, gap fixed-effects structure)\n")
cat(strrep("#", 70), "\n")

for (i in seq_along(groups)) {
  group <- groups[i]
  gname <- group_names[[i]]
  gdf <- droplevels(data[data$group == group, ])

  cat("\n", strrep("=", 60), "\n")
  cat(" ", gname, "RE selection (combined, n =", nrow(gdf),
      ", sentences =", nlevels(gdf$sentence), ")\n")
  cat(strrep("=", 60), "\n\n")

  result <- backward_eliminate_re(gdf, fe_gap)
  re_selected[["combined"]][[group]] <- result$selected_re
  re_comparisons[["combined"]][[group]] <- result

  cat(sprintf("\n  Selected RE for %s (combined): %s\n    (%s)\n",
              gname, result$selected_re, result$reason))
}

re_rows <- data.frame(
  group = groups,
  group_name = group_names[groups],
  selected_re = sapply(groups, function(g) re_selected[["combined"]][[g]]),
  selected_key = sapply(groups, function(g) re_comparisons[["combined"]][[g]]$selected_key),
  reason = sapply(groups, function(g) re_comparisons[["combined"]][[g]]$reason),
  stringsAsFactors = FALSE
)
re_rows$aic_selected <- sapply(groups, function(g) {
  key <- re_comparisons[["combined"]][[g]]$selected_key
  inf <- re_comparisons[["combined"]][[g]]$info[[key]]
  if (!is.null(inf) && !is.na(inf$aic)) inf$aic else NA
})
re_rows$aic_baseline <- sapply(groups, function(g) {
  inf <- re_comparisons[["combined"]][[g]]$info[["baseline"]]
  if (!is.null(inf) && !is.na(inf$aic)) inf$aic else NA
})
re_rows$singular_selected <- sapply(groups, function(g) {
  key <- re_comparisons[["combined"]][[g]]$selected_key
  inf <- re_comparisons[["combined"]][[g]]$info[[key]]
  if (!is.null(inf)) inf$singular else NA
})
re_csv_path <- file.path(output_dir, "results_combined_re_selection.csv")
write.csv(re_rows, re_csv_path, row.names = FALSE)
cat(sprintf("Saved %s\n", re_csv_path))


# ---------------------------------------------------------------------------
# Combined generalization-gap results
# ---------------------------------------------------------------------------

cat("\n", strrep("#", 70), "\n")
cat("  GENERALIZATION GAP (combined data only)\n")
cat(strrep("#", 70), "\n")

for (i in seq_along(groups)) {
  group <- groups[i]
  gname <- group_names[[i]]
  gdf <- droplevels(data[data$group == group, ])
  re <- re_selected[["combined"]][[group]]

  cat("\n", strrep("=", 60), "\n")
  cat(" ", gname, "gap analysis (n =", nrow(gdf),
      ", train =", sum(gdf$train_or_test == "train"),
      ", test =", sum(gdf$train_or_test == "test"), ")\n")
  cat("  RE:", re, "\n")
  cat(strrep("=", 60), "\n")

  m_add <- lmer(as.formula(paste0(
    "advantage ~ arch + entity + train_or_test + split + ", re)),
    data = gdf, REML = TRUE)
  m_arch_tot <- lmer(as.formula(paste0(
    "advantage ~ arch * train_or_test + entity + split + ", re)),
    data = gdf, REML = TRUE)
  m_ent_tot <- lmer(as.formula(paste0(
    "advantage ~ arch + entity * train_or_test + split + ", re)),
    data = gdf, REML = TRUE)
  m_both <- lmer(as.formula(paste0(
    "advantage ~ arch * entity * train_or_test + split + ", re)),
    data = gdf, REML = TRUE)

  cat("\nLRT vs additive:\n")
  comp_arch <- anova(m_add, m_arch_tot)
  comp_ent <- anova(m_add, m_ent_tot)
  p_arch <- comp_arch[2, "Pr(>Chisq)"]
  p_ent <- comp_ent[2, "Pr(>Chisq)"]
  cat(sprintf("  arch:train_or_test    Chisq=%.2f  df=%d  p=%s  %s\n",
              comp_arch[2, "Chisq"], comp_arch[2, "Df"],
              format_p(p_arch), sig_stars(p_arch)))
  cat(sprintf("  entity:train_or_test  Chisq=%.2f  df=%d  p=%s  %s\n",
              comp_ent[2, "Chisq"], comp_ent[2, "Df"],
              format_p(p_ent), sig_stars(p_ent)))

  aov <- anova(m_both)

  emm_arch <- emmeans(m_both, pairwise ~ arch | train_or_test, adjust = "tukey")
  arch_df <- merge_ci(
    as.data.frame(summary(emm_arch$contrasts)),
    emm_arch$contrasts
  )

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
  arch_ent_df <- merge_ci(
    as.data.frame(summary(emm_arch_ent$contrasts)),
    emm_arch_ent$contrasts
  )

  emm_ent_arch <- emmeans(
    m_both,
    pairwise ~ entity | train_or_test * arch,
    adjust = "none"
  )
  ent_arch_df <- summarize_familywise_contrasts(
    emm_ent_arch$contrasts,
    adjust = "mvt"
  )

  write.csv(as.data.frame(aov),
            file.path(output_dir, sprintf("results_gap_%s_anova.csv", group)),
            row.names = TRUE)
  write.csv(arch_df,
            file.path(output_dir, sprintf("results_gap_%s_arch_pairwise.csv", group)),
            row.names = FALSE)
  write.csv(ent_df,
            file.path(output_dir, sprintf("results_gap_%s_entity.csv", group)),
            row.names = FALSE)
  write.csv(arch_ent_df,
            file.path(output_dir, sprintf("results_gap_%s_arch_pairwise_by_entity.csv", group)),
            row.names = FALSE)
  write.csv(ent_arch_df,
            file.path(output_dir, sprintf("results_gap_%s_entity_by_arch.csv", group)),
            row.names = FALSE)

  cat(sprintf("Saved combined gap CSVs for %s\n", gname))
}

write_warning_log(output_dir, "results_gap_combined_warnings.txt")
cat("\nDone.\n")
