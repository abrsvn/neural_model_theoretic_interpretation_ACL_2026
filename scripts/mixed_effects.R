#!/usr/bin/env Rscript
#
# Mixed-effects appendix analysis
#
# Runs the appendix mixed-effects selection procedure
# and writes these appendix tables:
#   - stat_test_random_slopes.tex
#   - stat_train_random_slopes.tex
#   - stat_test_anova.tex
#   - stat_train_anova.tex
#   - stat_test_effect_sizes.tex
#   - stat_train_effect_sizes.tex
#   - stat_test_entity_effect.tex
#   - stat_train_entity_effect.tex
#   - stat_test_arch_pairwise.tex
#   - stat_train_arch_pairwise.tex
#
# It intentionally does NOT emit:
#   - combined / gap / per-split outputs
#   - residual / QQ diagnostic figures
#   - sentence-level appendix tables

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

options(width = 120, contrasts = c("contr.sum", "contr.poly"))
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 200000, pbkrtest.limit = 0)


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript mixed_effects.R <sentence_data.csv> <output_dir>\n")
  quit(status = 1)
}

data_file <- args[1]
latex_dir <- args[2]
dir.create(latex_dir, showWarnings = FALSE, recursive = TRUE)

script_dir <- dirname(
  sub("--file=", "", commandArgs()[grep("--file=", commandArgs())])
)


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

p_str_latex <- function(p) {
  ifelse(is.na(p), "NA",
    ifelse(p < 0.001, "$<$.001",
      ifelse(p < 0.01, sprintf("%.3f", p),
        sprintf("%.2f", p))))
}

esc <- function(s) gsub("_", "\\\\_", s)

find_anova_effect <- function(aov, eff) {
  if (eff %in% rownames(aov)) return(eff)
  parts <- strsplit(eff, ":")[[1]]
  n <- length(parts)
  if (n == 2) {
    perms <- list(parts, rev(parts))
  } else if (n == 3) {
    a <- parts[1]
    b <- parts[2]
    c <- parts[3]
    perms <- list(c(a, b, c), c(a, c, b), c(b, a, c),
                  c(b, c, a), c(c, a, b), c(c, b, a))
  } else {
    return(NA_character_)
  }
  for (perm in perms) {
    candidate <- paste(perm, collapse = ":")
    if (candidate %in% rownames(aov)) return(candidate)
  }
  NA_character_
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

format_ci <- function(lower, upper) {
  sprintf("[%.3f, %.3f]", lower, upper)
}

backward_eliminate_fe <- function(gdf, re_str) {
  all_interactions <- c("arch:entity:split",
                        "arch:entity", "arch:split", "entity:split")

  is_droppable <- function(term, active) {
    if (term == "arch:entity:split") return(TRUE)
    !("arch:entity:split" %in% active)
  }

  build_formula <- function(active_interactions) {
    fe <- "advantage ~ arch + entity + split"
    for (term in active_interactions) fe <- paste0(fe, " + ", term)
    paste0(fe, " + ", re_str)
  }

  make_key <- function(terms) {
    if (length(terms) == 0) "additive"
    else paste(sort(terms), collapse = "+")
  }

  make_label <- function(terms) {
    if (length(terms) == 0) "arch + entity + split"
    else if (length(terms) == 4) "arch * entity * split"
    else paste0("arch + entity + split + ", paste(terms, collapse = " + "))
  }

  try_fit <- function(formula_str) {
    tryCatch(
      lmer(as.formula(formula_str), data = gdf, REML = TRUE),
      error = function(e) NULL
    )
  }

  all_fits <- list()
  all_labels <- list()
  all_comparisons <- list()
  path <- character(0)

  current_terms <- all_interactions
  current_key <- make_key(current_terms)
  current_fit <- try_fit(build_formula(current_terms))

  if (is.null(current_fit)) {
    cat("  Full model failed, falling back to additive\n")
    additive_fit <- try_fit(build_formula(character(0)))
    all_fits[["additive"]] <- additive_fit
    all_labels[["additive"]] <- make_label(character(0))
    return(list(
      fit = additive_fit,
      terms = character(0),
      label = "arch + entity + split",
      reason = "full model failed",
      all_fits = all_fits,
      all_labels = all_labels,
      comparisons = all_comparisons,
      path = "additive",
      selected_key = "additive"
    ))
  }

  all_fits[[current_key]] <- current_fit
  all_labels[[current_key]] <- make_label(current_terms)
  path <- c(path, current_key)

  cat("\n  Backward FE elimination from: arch * entity * split\n")

  repeat {
    if (length(current_terms) == 0) break

    droppable <- Filter(function(t) is_droppable(t, current_terms), current_terms)
    if (length(droppable) == 0) break

    drop_candidates <- list()
    for (term in droppable) {
      reduced_terms <- setdiff(current_terms, term)
      reduced_fit <- try_fit(build_formula(reduced_terms))
      if (is.null(reduced_fit)) next

      reduced_key <- make_key(reduced_terms)
      if (is.null(all_fits[[reduced_key]])) {
        all_fits[[reduced_key]] <- reduced_fit
        all_labels[[reduced_key]] <- make_label(reduced_terms)
      }

      comp <- tryCatch(anova(reduced_fit, current_fit), error = function(e) NULL)
      if (is.null(comp)) next

      pv <- comp[2, "Pr(>Chisq)"]
      cat(sprintf("    drop %-20s: Chisq=%.2f df=%d p=%s %s\n",
                  term, comp[2, "Chisq"], comp[2, "Df"],
                  format_p(pv), sig_stars(pv)))

      all_comparisons[[length(all_comparisons) + 1]] <- list(
        from_key = current_key,
        to_key = reduced_key,
        chisq = comp[2, "Chisq"],
        df = comp[2, "Df"],
        p = pv,
        term_dropped = term
      )

      drop_candidates[[term]] <- list(
        term = term,
        p = pv,
        fit = reduced_fit,
        reduced_terms = reduced_terms,
        reduced_key = reduced_key
      )
    }

    if (length(drop_candidates) == 0) break

    non_sig <- Filter(function(d) d$p >= 0.05, drop_candidates)
    if (length(non_sig) == 0) {
      cat(sprintf("    All droppable terms significant (%d tested) -> stop\n",
                  length(drop_candidates)))
      break
    }

    best <- non_sig[[which.max(sapply(non_sig, function(d) d$p))]]
    cat(sprintf("    Dropping '%s' (p=%.3f)\n", best$term, best$p))
    current_terms <- best$reduced_terms
    current_key <- best$reduced_key
    current_fit <- best$fit
    path <- c(path, current_key)
  }

  if (is.null(all_fits[["additive"]])) {
    additive_fit <- try_fit(build_formula(character(0)))
    if (!is.null(additive_fit)) {
      all_fits[["additive"]] <- additive_fit
      all_labels[["additive"]] <- make_label(character(0))
    }
  }

  label <- make_label(current_terms)
  cat(sprintf("  SELECTED FE: %s\n", label))

  list(
    fit = current_fit,
    terms = current_terms,
    label = label,
    reason = sprintf("backward elimination, %d interaction(s) retained",
                     length(current_terms)),
    all_fits = all_fits,
    all_labels = all_labels,
    comparisons = all_comparisons,
    path = path,
    selected_key = current_key
  )
}

backward_eliminate_re <- function(gdf, fe_str) {
  comp_formulas <- c(
    ent = "(0+entity|sentence)",
    rec = "(0+is_recurrent|sentence)",
    splt = "(0+split|sentence)",
    arch = "(1|arch:sentence)"
  )

  build_re <- function(active) {
    parts <- "(1|sentence)"
    for (nm in active) parts <- paste0(parts, " + ", comp_formulas[nm])
    paste0(parts, " + (1|model_id)")
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
  slope_terms <- intersect(current_terms, c("ent", "rec", "splt"))

  if (length(slope_terms) >= 1) {
    slope_map <- c(ent = "entity", rec = "is_recurrent", splt = "split")
    slope_vars <- unname(slope_map[slope_terms])
    cov_part <- paste0("(1 + ", paste(slope_vars, collapse = " + "), " | sentence)")
    other <- c()
    if ("arch" %in% current_terms) other <- c(other, "(1|arch:sentence)")
    other <- c(other, "(1|model_id)")
    cov_re <- paste(c(cov_part, other), collapse = " + ")

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

longtable_wrap <- function(col_spec, header_line, caption, label) {
  ncols <- nchar(gsub("[^lrcLRC]", "", col_spec))
  head <- c(
    "\\scriptsize",
    sprintf("\\begin{longtable}{%s}", col_spec),
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s} \\\\", label),
    "\\toprule",
    paste0(header_line, " \\\\"),
    "\\midrule",
    "\\endfirsthead",
    "",
    sprintf("\\multicolumn{%d}{c}{\\scriptsize\\textit{(continued)}} \\\\", ncols),
    "\\toprule",
    paste0(header_line, " \\\\"),
    "\\midrule",
    "\\endhead",
    "",
    sprintf("\\midrule \\multicolumn{%d}{r@{}}{\\scriptsize\\textit{(continued on next page)}} \\\\", ncols),
    "\\endfoot",
    "",
    "\\bottomrule",
    "\\endlastfoot"
  )
  foot <- c(
    "\\end{longtable}",
    "\\normalsize"
  )
  list(head = head, foot = foot)
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

# Group SRN/GRU/LSTM as "recurrent" and all other architectures as "attention".
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


# ===========================================================================
# Random effects selection (test + train, full fixed-effects structure)
# ===========================================================================

re_selected <- list(test = list(), train = list())
re_comparisons <- list(test = list(), train = list())

fe_full <- "advantage ~ arch * entity * split"

for (tot in c("test", "train")) {
  cat("\n", strrep("#", 70), "\n")
  cat("  Random effects selection (", tot,
      "sentences, full fixed-effects structure)\n")
  cat(strrep("#", 70), "\n")

  for (i in seq_along(groups)) {
    group <- groups[i]
    gname <- group_names[[i]]
    gdf <- droplevels(data[data$group == group & data$train_or_test == tot, ])

    cat("\n", strrep("=", 60), "\n")
    cat(" ", gname, "RE selection (", tot, ", n =", nrow(gdf),
        ", sentences =", nlevels(gdf$sentence), ")\n")
    cat(strrep("=", 60), "\n\n")

    result <- backward_eliminate_re(gdf, fe_full)
    re_selected[[tot]][[group]] <- result$selected_re
    re_comparisons[[tot]][[group]] <- result

    cat(sprintf("\n  Selected RE for %s (%s): %s\n    (%s)\n",
                gname, tot, result$selected_re, result$reason))
  }

  cat(sprintf("\n\nRandom-effects summary (%s):\n", tot))
  for (i in seq_along(groups)) {
    cat(sprintf("  %-15s: %s\n", group_names[[i]], re_selected[[tot]][[groups[i]]]))
  }
}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

results <- list(test = list(), train = list())


# ---------------------------------------------------------------------------
# Main analysis loop (fixed-effects selection with the chosen random effects)
# ---------------------------------------------------------------------------

for (tot in c("test", "train")) {
  cat("\n", strrep("#", 70), "\n")
  cat("  ", toupper(tot),
      "sentences (fixed-effects selection, locked random effects)\n")
  cat(strrep("#", 70), "\n")

  for (i in seq_along(groups)) {
    group <- groups[i]
    gname <- group_names[[i]]
    gdf <- droplevels(data[data$group == group & data$train_or_test == tot, ])

    n_sent <- nlevels(gdf$sentence)
    n_model <- nlevels(gdf$model_id)
    re <- re_selected[[tot]][[group]]

    cat("\n", strrep("=", 70), "\n")
    cat(" ", gname, " [", tot, "] (n =", nrow(gdf),
        ", sentences =", n_sent, ", models =", n_model, ")\n")
    cat("  RE:", re, "\n")
    cat(strrep("=", 70), "\n")

    fe_result <- backward_eliminate_fe(gdf, re)
    model <- fe_result$fit
    if (is.null(model)) {
      cat("  All models failed for", gname, tot, "\n")
      next
    }

    cat("\nSelected model:", fe_result$label, "\n")

    cat("\nANOVA (Type III):\n")
    aov <- anova(model)
    print(aov)

    cat("\nRandom effects:\n")
    print(VarCorr(model))
    cat("Residual SD:", sigma(model), "\n")

    cat("\nArchitecture pairwise (averaged over entity, Tukey):\n")
    emm_arch <- emmeans(model, pairwise ~ arch, adjust = "tukey")
    arch_avg_df <- merge_ci(as.data.frame(summary(emm_arch$contrasts)),
                            emm_arch$contrasts)
    print(arch_avg_df)

    cat("\nArchitecture pairwise by entity (Tukey):\n")
    emm_arch_ent <- emmeans(model, pairwise ~ arch | entity, adjust = "tukey")
    arch_ent_df <- merge_ci(as.data.frame(summary(emm_arch_ent$contrasts)),
                            emm_arch_ent$contrasts)
    print(arch_ent_df)

    cat("\nEntity effect per architecture:\n")
    emm_ent <- emmeans(model, pairwise ~ entity | arch, adjust = "none")
    ent_arch_df <- summarize_familywise_contrasts(
      emm_ent$contrasts,
      adjust = "mvt"
    )
    print(ent_arch_df)

    results[[tot]][[group]] <- list(
      fe_result = fe_result,
      re_str = re,
      model = model,
      sigma_resid = sigma(model),
      anova = aov,
      arch_avg = arch_avg_df,
      arch_ent = arch_ent_df,
      ent_arch = ent_arch_df
    )

    write.csv(as.data.frame(aov),
              file.path(latex_dir, sprintf("results_%s_%s_anova.csv", tot, group)),
              row.names = TRUE)
    write.csv(arch_avg_df,
              file.path(latex_dir, sprintf("results_%s_%s_arch_pairwise.csv", tot, group)),
              row.names = FALSE)
    write.csv(arch_ent_df,
              file.path(latex_dir, sprintf("results_%s_%s_arch_by_entity.csv", tot, group)),
              row.names = FALSE)
    write.csv(ent_arch_df,
              file.path(latex_dir, sprintf("results_%s_%s_entity_effect.csv", tot, group)),
              row.names = FALSE)
  }
}

for (tot in c("test", "train")) {
  re_rows <- data.frame(
    group = groups,
    group_name = group_names[groups],
    selected_re = sapply(groups, function(g) re_selected[[tot]][[g]]),
    selected_key = sapply(groups, function(g) re_comparisons[[tot]][[g]]$selected_key),
    reason = sapply(groups, function(g) re_comparisons[[tot]][[g]]$reason),
    selected_fe_key = sapply(groups, function(g) {
      res <- results[[tot]][[g]]
      if (is.null(res)) return(NA_character_)
      res$fe_result$selected_key
    }),
    selected_fe_label = sapply(groups, function(g) {
      res <- results[[tot]][[g]]
      if (is.null(res)) return(NA_character_)
      res$fe_result$label
    }),
    stringsAsFactors = FALSE
  )
  re_rows$aic_selected <- sapply(groups, function(g) {
    key <- re_comparisons[[tot]][[g]]$selected_key
    inf <- re_comparisons[[tot]][[g]]$info[[key]]
    if (!is.null(inf) && !is.na(inf$aic)) inf$aic else NA
  })
  re_rows$aic_baseline <- sapply(groups, function(g) {
    inf <- re_comparisons[[tot]][[g]]$info[["baseline"]]
    if (!is.null(inf) && !is.na(inf$aic)) inf$aic else NA
  })
  re_rows$singular_selected <- sapply(groups, function(g) {
    key <- re_comparisons[[tot]][[g]]$selected_key
    inf <- re_comparisons[[tot]][[g]]$info[[key]]
    if (!is.null(inf)) inf$singular else NA
  })
  csv_path <- file.path(latex_dir, sprintf("results_%s_re_selection.csv", tot))
  write.csv(re_rows, csv_path, row.names = FALSE)
  cat(sprintf("Updated %s with FE selections\n", csv_path))
}


# ===========================================================================
# LaTeX tables + trimmed random-slopes diagnostics
# ===========================================================================

source(file.path(script_dir, "mixed_effects_latex.R"))
source(file.path(script_dir, "mixed_effects_diagnostics.R"))

cat("\nDone.\n")
