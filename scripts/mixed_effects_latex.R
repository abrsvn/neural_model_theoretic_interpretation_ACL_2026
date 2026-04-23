# mixed_effects_latex.R -- trimmed LaTeX table generation
# Sourced from mixed_effects.R. Parent provides:
#   results, groups, group_names, latex_dir, re_selected, re_comparisons
#   longtable_wrap(), esc(), p_str_latex(), format_ci(), merge_ci(),
#   format_p(), sig_stars(), find_anova_effect()

cat("\n\nGenerating trimmed LaTeX tables...\n")

detect_cols <- function(df) {
  t_col <- intersect(c("t.ratio", "z.ratio", "statistic", "t.value"), names(df))
  p_col <- intersect(c("p.value", "Pr...t..", "adj.p.value"), names(df))
  list(
    est = "estimate",
    se = "SE",
    t = if (length(t_col) > 0) t_col[1] else NA,
    df = if ("df" %in% names(df)) "df" else NA,
    p = if (length(p_col) > 0) p_col[1] else NA
  )
}

extract_row_values <- function(r, cols, sigma_resid, negate = FALSE) {
  est <- if (negate) -r[["estimate"]] else r[["estimate"]]
  tval <- if (!is.na(cols$t)) {
    if (negate) -r[[cols$t]] else r[[cols$t]]
  } else NA
  dfv <- if (!is.na(cols$df)) r[[cols$df]] else NA
  pval <- if (!is.na(cols$p)) r[[cols$p]] else NA

  ci_str <- if (negate) {
    format_ci(-r[["upper.CL"]], -r[["lower.CL"]])
  } else {
    format_ci(r[["lower.CL"]], r[["upper.CL"]])
  }

  list(
    est = est,
    ci_str = ci_str,
    se = r[["SE"]],
    t_str = if (!is.na(tval)) sprintf("%.2f", tval) else "--",
    df_str = if (!is.na(dfv)) sprintf("%.1f", dfv) else "--",
    p_str = if (!is.na(pval)) p_str_latex(pval) else "--",
    d_val = est / sigma_resid
  )
}

format_contrast_row <- function(labels, vals) {
  label_str <- paste(labels, collapse = " & ")
  sprintf(
    "%s & %+.3f & %s & %.3f & %s & %s & %s & %.2f \\\\",
    label_str,
    vals$est, vals$ci_str, vals$se,
    vals$t_str, vals$df_str, vals$p_str,
    vals$d_val
  )
}


# ---------------------------------------------------------------------------
# Table: ANOVA from selected model
# ---------------------------------------------------------------------------

effect_display_map <- c(
  arch = "Architecture",
  entity = "Entity vectors",
  split = "Split",
  "arch:entity" = "Arch $\\times$ Entity",
  "entity:arch" = "Arch $\\times$ Entity",
  "arch:split" = "Arch $\\times$ Split",
  "split:arch" = "Arch $\\times$ Split",
  "entity:split" = "Entity $\\times$ Split",
  "split:entity" = "Entity $\\times$ Split",
  "arch:entity:split" = "Arch $\\times$ Entity $\\times$ Split",
  "arch:split:entity" = "Arch $\\times$ Entity $\\times$ Split",
  "entity:arch:split" = "Arch $\\times$ Entity $\\times$ Split",
  "entity:split:arch" = "Arch $\\times$ Entity $\\times$ Split",
  "split:arch:entity" = "Arch $\\times$ Entity $\\times$ Split",
  "split:entity:arch" = "Arch $\\times$ Entity $\\times$ Split"
)

for (tot in c("test", "train")) {
  lines <- c(
    "\\begin{table}[t]",
    "\\centering\\small",
    "\\begin{tabular}{@{}llrrrr@{}}",
    "\\toprule",
    "\\textbf{Test Group} & \\textbf{Effect} & $F$ & \\textbf{df} & $p$ & $\\eta^2_p$ \\\\",
    "\\midrule"
  )

  for (i in seq_along(groups)) {
    res <- results[[tot]][[groups[i]]]
    if (is.null(res)) next
    aov <- res$anova

    first <- TRUE
    for (eff in rownames(aov)) {
      display <- effect_display_map[eff]
      if (is.na(display)) display <- esc(eff)

      f_val <- aov[eff, "F value"]
      df1 <- aov[eff, "NumDF"]
      df2 <- aov[eff, "DenDF"]
      p_val <- aov[eff, "Pr(>F)"]
      eta2 <- (f_val * df1) / (f_val * df1 + df2)

      glabel <- if (first) group_names[[i]] else ""
      first <- FALSE
      df_str <- sprintf("%.0f, %.1f", df1, df2)

      lines <- c(lines, sprintf("%s & %s & %.2f & %s & %s & %.3f \\\\",
                                glabel, display,
                                f_val, df_str, p_str_latex(p_val), eta2))
    }
    if (i < length(groups)) lines <- c(lines, "\\addlinespace")
  }

  sel_labels <- c()
  for (i in seq_along(groups)) {
    res <- results[[tot]][[groups[i]]]
    if (!is.null(res)) {
      sel_labels <- c(sel_labels,
                      sprintf("%s: %s", group_names[[i]], res$fe_result$label))
    }
  }
  sel_note <- paste(sel_labels, collapse = "; ")
  lines <- c(lines,
    "\\bottomrule",
    "\\end{tabular}",
    sprintf(paste0("\\caption{Type III ANOVA for %s sentences ",
                   "(Satterthwaite df). ",
                   "Selected fixed-effects structure per group (backward elimination): %s. ",
                   "Random-effects structure selected separately per group ",
                   "(see Table~\\ref{tab:%s_random_slopes}).}"), tot, sel_note, tot),
    sprintf("\\label{tab:%s_anova}", tot),
    "\\end{table}"
  )

  fpath <- file.path(latex_dir, sprintf("stat_%s_anova.tex", tot))
  writeLines(lines, fpath)
  cat("Wrote", fpath, "\n")
}


# ---------------------------------------------------------------------------
# Table: Architecture pairwise by entity condition
# ---------------------------------------------------------------------------

for (tot in c("test", "train")) {
  cap <- sprintf(paste0("Pairwise architecture comparisons for %s ",
                        "sentences by entity condition (Tukey adjustment). ",
                        "Positive: first architecture has higher advantage."), tot)
  lab <- sprintf("tab:%s_arch_pairwise", tot)
  hdr <- paste0("\\textbf{Test Group} & \\textbf{Entity} & \\textbf{Comparison} & ",
                "\\textbf{Est.} & \\textbf{[95\\% CI]} & \\textbf{SE} & $t$ & ",
                "\\textbf{df} & $p_{\\text{Tukey}}$ & $d$")
  lt <- longtable_wrap("@{}lllrcrrrrr@{}", hdr, cap, lab)
  lines <- lt$head

  for (i in seq_along(groups)) {
    res <- results[[tot]][[groups[i]]]
    if (is.null(res)) next
    cdf <- res$arch_ent
    cols <- detect_cols(cdf)

    entity_levels <- c("noent", "ent")
    for (ei in seq_along(entity_levels)) {
      ent_val <- entity_levels[ei]
      ent_rows <- cdf[cdf$entity == ent_val, ]
      ent_label <- if (ent_val == "noent") "$-$ent" else "$+$ent"

      first <- TRUE
      for (j in seq_len(nrow(ent_rows))) {
        r <- ent_rows[j, ]
        glabel <- if (first && ei == 1) group_names[[i]] else ""
        elabel <- if (first) ent_label else ""
        first <- FALSE

        vals <- extract_row_values(r, cols, res$sigma_resid)
        lines <- c(lines, format_contrast_row(
          c(glabel, elabel, esc(as.character(r[["contrast"]]))), vals))
      }
      lines <- c(lines, "\\addlinespace[2pt]")
    }
    if (i < length(groups)) lines <- c(lines, "\\addlinespace")
  }

  lines <- c(lines, lt$foot)
  fpath <- file.path(latex_dir, sprintf("stat_%s_arch_pairwise.tex", tot))
  writeLines(lines, fpath)
  cat("Wrote", fpath, "\n")
}


# ---------------------------------------------------------------------------
# Table: Entity effect per architecture
# ---------------------------------------------------------------------------

for (tot in c("test", "train")) {
  lines <- c(
    "\\begin{table}[t]",
    "\\centering\\small",
    "\\begin{tabular}{@{}llrcrrrrr@{}}",
    "\\toprule",
    paste0("\\textbf{Test Group} & \\textbf{Architecture} & ",
           "\\textbf{Ent effect} & \\textbf{[95\\% CI]} & \\textbf{SE} & $t$ & ",
           "\\textbf{df} & $p$ & $d$ \\\\"),
    "\\midrule"
  )

  has_collapsed <- FALSE
  for (i in seq_along(groups)) {
    res <- results[[tot]][[groups[i]]]
    if (is.null(res)) next
    cdf <- res$ent_arch
    cols <- detect_cols(cdf)

    has_arch_entity <- any(c("arch:entity", "arch:entity:split") %in%
                             res$fe_result$terms)
    is_constant <- !has_arch_entity || sd(cdf[["estimate"]]) < 1e-10

    if (is_constant) {
      has_collapsed <- TRUE
      vals <- extract_row_values(cdf[1, ], cols, res$sigma_resid, negate = TRUE)
      lines <- c(lines, format_contrast_row(c(group_names[[i]], "All"), vals))
    } else {
      first <- TRUE
      for (j in seq_len(nrow(cdf))) {
        r <- cdf[j, ]
        glabel <- if (first) group_names[[i]] else ""
        first <- FALSE

        vals <- extract_row_values(r, cols, res$sigma_resid, negate = TRUE)
        arch_col <- intersect(c("arch", "by"), names(cdf))
        arch_name <- if (length(arch_col) > 0) as.character(r[[arch_col[1]]]) else "?"

        lines <- c(lines, format_contrast_row(
          c(glabel, esc(arch_name)), vals))
      }
    }
    if (i < length(groups)) lines <- c(lines, "\\addlinespace")
  }

  collapse_note <- if (has_collapsed) {
    " Where the selected model includes no architecture $\\times$ entity interaction, a single pooled row (\\texttt{All}) is shown."
  } else {
    ""
  }
  lines <- c(lines,
    "\\bottomrule",
    "\\end{tabular}",
    sprintf(paste0("\\caption{Entity vector effect per architecture for %s ",
                   "sentences (ent $-$ noent). Positive: entity vectors ",
                   "improve advantage. $mvt$-adjusted simultaneous $p$ values ",
                   "and confidence intervals.%s}"),
            tot, collapse_note),
    sprintf("\\label{tab:%s_entity_effect}", tot),
    "\\end{table}"
  )

  fpath <- file.path(latex_dir, sprintf("stat_%s_entity_effect.tex", tot))
  writeLines(lines, fpath)
  cat("Wrote", fpath, "\n")
}


# ---------------------------------------------------------------------------
# EFFECT SIZE SUMMARY TABLE
# ---------------------------------------------------------------------------

cat("\nGenerating effect size summaries...\n")

compute_eta2 <- function(aov, eff) {
  actual <- find_anova_effect(aov, eff)
  if (is.na(actual)) return(NA)
  f_val <- aov[actual, "F value"]
  df1 <- aov[actual, "NumDF"]
  df2 <- aov[actual, "DenDF"]
  (f_val * df1) / (f_val * df1 + df2)
}

for (tot in c("test", "train")) {
  lines <- c(
    "\\begin{table}[t]",
    "\\centering\\scriptsize",
    "\\begin{tabular}{@{}lrrrrrrrrr@{}}",
    "\\toprule",
    paste0("\\textbf{Test Group} & \\textbf{Max $|d|$} & ",
           "\\textbf{Mean ent.\\ $d$} & ",
           "$\\eta^2_p$(arch) & $\\eta^2_p$(ent) & ",
           "$\\eta^2_p$(split) & $\\eta^2_p$(A$\\times$E) & ",
           "$\\eta^2_p$(A$\\times$S) & $\\eta^2_p$(E$\\times$S) & ",
           "$\\eta^2_p$(3-way) \\\\"),
    "\\midrule"
  )

  es_rows <- list()
  for (i in seq_along(groups)) {
    res <- results[[tot]][[groups[i]]]
    if (is.null(res)) next

    sig <- res$sigma_resid
    aov <- res$anova
    ent_rows <- res$arch_ent[res$arch_ent$entity == "ent", ]
    max_d <- max(abs(ent_rows[["estimate"]] / sig))
    mean_ent_d <- mean(-res$ent_arch[["estimate"]] / sig)

    effects <- c("arch", "entity", "split",
                 "arch:entity", "arch:split", "entity:split",
                 "arch:entity:split")
    eta2_vals <- sapply(effects, function(e) compute_eta2(aov, e))
    eta_strs <- ifelse(is.na(eta2_vals), "--", sprintf("%.3f", eta2_vals))

    lines <- c(lines, sprintf(
      "%s & %.2f & %.2f & %s & %s & %s & %s & %s & %s & %s \\\\",
      group_names[[i]], max_d, mean_ent_d,
      eta_strs[1], eta_strs[2], eta_strs[3],
      eta_strs[4], eta_strs[5], eta_strs[6], eta_strs[7]
    ))

    es_rows[[length(es_rows) + 1]] <- data.frame(
      group = groups[i],
      group_name = group_names[[i]],
      max_abs_d = max_d,
      mean_entity_d = mean_ent_d,
      eta2_arch = eta2_vals["arch"],
      eta2_entity = eta2_vals["entity"],
      eta2_split = eta2_vals["split"],
      eta2_arch_entity = eta2_vals["arch:entity"],
      eta2_arch_split = eta2_vals["arch:split"],
      eta2_entity_split = eta2_vals["entity:split"],
      eta2_3way = eta2_vals["arch:entity:split"],
      stringsAsFactors = FALSE
    )
  }

  lines <- c(lines,
    "\\bottomrule",
    "\\end{tabular}",
    sprintf(paste0("\\caption{Effect size summary for %s sentences. ",
                   "Max $|d|$: largest Cohen's $d$ among architecture pairwise ",
                   "comparisons ($+$ent condition). ",
                   "Mean ent.\\ $d$: mean entity benefit $d$ across architectures. ",
                   "$\\eta^2_p$: partial eta-squared from Type III ANOVA ",
                   "(\\texttt{--} = interaction not in selected model).}"), tot),
    sprintf("\\label{tab:%s_effect_sizes}", tot),
    "\\end{table}"
  )

  fpath <- file.path(latex_dir, sprintf("stat_%s_effect_sizes.tex", tot))
  writeLines(lines, fpath)
  cat("Wrote", fpath, "\n")

  if (length(es_rows) > 0) {
    write.csv(do.call(rbind, es_rows),
              file.path(latex_dir, sprintf("results_%s_effect_sizes.csv", tot)),
              row.names = FALSE)
  }
}
