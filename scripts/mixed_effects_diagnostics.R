# mixed_effects_diagnostics.R -- trimmed random-slopes tables only
# Sourced from mixed_effects.R. Parent provides:
#   groups, group_names, re_comparisons, latex_dir, longtable_wrap(), p_str_latex()

cat("\nGenerating trimmed random-slopes tables...\n")

re_term_order <- c("ent", "rec", "splt", "arch")
re_term_display <- c(
  ent = "entity slope",
  rec = "is\\_recurrent slope",
  splt = "split slope",
  arch = "arch:sentence"
)

format_re_label <- function(key) {
  if (key == "baseline") return("baseline")

  use_cov <- grepl("\\+cov$", key)
  base_key <- sub("\\+cov$", "", key)
  terms <- strsplit(base_key, "\\+", perl = TRUE)[[1]]
  terms <- re_term_order[re_term_order %in% terms]
  label <- paste0("+ ", paste(unname(re_term_display[terms]), collapse = " + "))

  if (use_cov) {
    paste0(label, " +cov")
  } else {
    label
  }
}

ordered_re_keys <- function(info) {
  base_keys <- c("baseline")
  for (k in seq_along(re_term_order)) {
    combos <- combn(re_term_order, k, simplify = FALSE)
    base_keys <- c(
      base_keys,
      vapply(combos, function(combo) paste(sort(combo), collapse = "+"), FUN.VALUE = character(1))
    )
  }

  display_keys <- intersect(base_keys, names(info))
  cov_keys <- intersect(paste0(base_keys[base_keys != "baseline"], "+cov"), names(info))
  c(display_keys, cov_keys)
}

for (tot in c("test", "train")) {
  cap <- sprintf(paste0("Random effects selection for %s sentences. ",
                        "All RE structures compared to baseline (1$|$sentence) + (1$|$model) ",
                        "via LRT using the full FE model (arch $\\times$ entity $\\times$ split). ",
                        "Backward elimination from richest non-singular model; ",
                        "selected RE per group shown in bold. ",
                        "$\\Delta$AIC: negative = better than baseline. ",
                        "\\texttt{singular}: random-effect covariance estimated at boundary. ",
                        "Covariance upgrade (+cov) tested for final slope terms."), tot)
  hdr <- paste0("\\textbf{Test Group} & \\textbf{RE Structure} & ",
                "\\textbf{AIC} & $\\Delta$\\textbf{AIC} & ",
                "$\\chi^2$ & \\textbf{df} & $p$ & \\textbf{Note}")
  lt <- longtable_wrap("@{}llrrrrrl@{}", hdr, cap,
                       sprintf("tab:%s_random_slopes", tot))
  lines <- lt$head

  for (i in seq_along(groups)) {
    group <- groups[i]
    rc <- re_comparisons[[tot]][[group]]
    if (is.null(rc)) next

    info <- rc$info
    lrt <- rc$lrt_vs_base
    sel_key <- rc$selected_key
    aic_base <- if (!is.null(info[["baseline"]]) && info[["baseline"]]$converged) {
      info[["baseline"]]$aic
    } else {
      NA
    }

    display_keys <- ordered_re_keys(info)

    first <- TRUE
    for (key in display_keys) {
      mi <- info[[key]]
      if (is.null(mi)) next
      glabel <- if (first) group_names[[i]] else ""
      first <- FALSE

      dlabel <- format_re_label(key)

      if (!mi$converged) {
        lines <- c(lines, sprintf(
          "%s & %s & -- & -- & -- & -- & -- & failed \\\\",
          glabel, dlabel
        ))
        next
      }

      aic_str <- sprintf("%.1f", mi$aic)
      daic_str <- if (!is.na(aic_base) && key != "baseline") {
        sprintf("%.1f", mi$aic - aic_base)
      } else {
        "--"
      }

      lr <- lrt[[key]]
      if (!is.null(lr)) {
        chisq_str <- sprintf("%.2f", lr$chisq)
        df_str <- sprintf("%d", lr$df)
        p_str <- p_str_latex(lr$p)
      } else {
        chisq_str <- "--"
        df_str <- "--"
        p_str <- "--"
      }

      note <- ""
      if (!is.na(mi$singular) && mi$singular) {
        note <- "singular"
      } else if (key == sel_key) {
        note <- "\\textbf{selected}"
      } else if (!is.null(lr) && !is.na(lr$p) && lr$p < 0.05) {
        note <- "improved"
      }

      lines <- c(lines, sprintf(
        "%s & %s & %s & %s & %s & %s & %s & %s \\\\",
        glabel, dlabel, aic_str, daic_str, chisq_str, df_str, p_str, note
      ))
    }
    if (i < length(groups)) lines <- c(lines, "\\addlinespace")
  }

  lines <- c(lines, lt$foot)

  fpath <- file.path(latex_dir, sprintf("stat_%s_random_slopes.tex", tot))
  writeLines(lines, fpath)
  cat("Wrote", fpath, "\n")
}
