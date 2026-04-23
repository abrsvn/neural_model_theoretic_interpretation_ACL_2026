#!/usr/bin/env Rscript
#
# Mixed-effects sentence-difficulty and architecture-gap diagnostics
#
# Refits the already-selected final per-group test/train models and emits:
#   - stat_test_sentence_difficulty.tex
#   - stat_train_sentence_difficulty.tex
#   - stat_test_sentence_arch_gap.tex
#   - stat_train_sentence_arch_gap.tex
#   - results_test_sentence_difficulty.csv
#   - results_train_sentence_difficulty.csv
#   - results_test_sentence_arch_gap.csv
#   - results_train_sentence_arch_gap.csv
#
# It intentionally does NOT emit:
#   - residual / QQ diagnostic PNGs
#   - combined gap outputs

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

options(width = 120, nwarnings = 10000, contrasts = c("contr.sum", "contr.poly"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript mixed_effects_sentence_diagnostics.R <sentence_data.csv> <output_dir>\n")
  quit(status = 1)
}

data_file <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

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

esc <- function(s) gsub("_", "\\\\_", s)

build_selected_formula <- function(selected_fe_key, re_str) {
  terms <- c("arch", "entity", "split")
  if (!is.na(selected_fe_key) && selected_fe_key != "additive") {
    terms <- c(terms, strsplit(selected_fe_key, "\\+", perl = TRUE)[[1]])
  }
  paste0("advantage ~ ", paste(terms, collapse = " + "), " + ", re_str)
}

load_selection_summary <- function(output_dir, phase) {
  path <- file.path(output_dir, sprintf("results_%s_re_selection.csv", phase))
  if (!file.exists(path)) {
    stop(sprintf("Selection summary does not exist: %s", path))
  }

  rows <- read.csv(path, stringsAsFactors = FALSE)
  if (!"selected_fe_key" %in% names(rows)) {
    stop(sprintf("Selection summary is missing selected_fe_key: %s", path))
  }

  split(rows, rows$group)
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

trunc_sent <- function(s, maxlen = 50) {
  ifelse(nchar(s) > maxlen, paste0(substr(s, 1, maxlen), "..."), s)
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

selection_summaries <- list(
  test = load_selection_summary(output_dir, "test"),
  train = load_selection_summary(output_dir, "train")
)
results <- list(test = list(), train = list())

for (tot in c("test", "train")) {
  cat("\n", strrep("#", 70), "\n")
  cat("  REFIT SELECTED MODELS (", tot, "sentences)\n")
  cat(strrep("#", 70), "\n")

  for (i in seq_along(groups)) {
    group <- groups[i]
    gname <- group_names[[i]]
    gdf <- droplevels(data[data$group == group & data$train_or_test == tot, ])
    selection_row <- selection_summaries[[tot]][[group]]
    if (is.null(selection_row) || nrow(selection_row) != 1) {
      stop(sprintf("Expected one selection row for %s %s", gname, tot))
    }

    re <- selection_row$selected_re[1]
    fe_key <- selection_row$selected_fe_key[1]
    formula_str <- build_selected_formula(fe_key, re)

    cat("\n", strrep("=", 70), "\n")
    cat(" ", gname, " [", tot, "] (n =", nrow(gdf),
        ", sentences =", nlevels(gdf$sentence), ", models =", nlevels(gdf$model_id), ")\n")
    cat("  RE:", re, "\n")
    cat("  FE key:", fe_key, "\n")
    cat(strrep("=", 70), "\n")

    model <- lmer(as.formula(formula_str), data = gdf, REML = TRUE)
    results[[tot]][[group]] <- list(
      re_str = re,
      fe_key = fe_key,
      model = model
    )
  }
}


for (tot in c("test", "train")) {
  cat(sprintf("\nSentence difficulty ranking (%s, random intercepts)...\n", tot))

  all_difficulty_rows <- list()
  cap <- sprintf(paste0("Sentence difficulty ranking from random intercepts (%s sentences). ",
                        "Top 10 hardest (most negative) and top 10 easiest (most positive) ",
                        "per test group. Model: selected FE + selected RE per group."), tot)
  hdr <- paste0("\\textbf{Test Group} & \\textbf{Rank} & ",
                "\\textbf{Sentence} & \\textbf{Intercept}")
  lt <- longtable_wrap("@{}lrlr@{}", hdr, cap,
                       sprintf("tab:%s_sentence_difficulty", tot))
  lines <- lt$head

  for (i in seq_along(groups)) {
    group <- groups[i]
    gname <- group_names[[i]]
    res <- results[[tot]][[group]]
    if (is.null(res)) next

    re <- ranef(res$model)
    sent_int <- data.frame(
      group = group,
      group_name = gname,
      sentence = unname(rownames(re$sentence)),
      intercept = unname(re$sentence[, "(Intercept)"]),
      stringsAsFactors = FALSE
    )
    sent_int <- sent_int[order(sent_int$intercept), ]
    sent_int$hard_rank <- seq_len(nrow(sent_int))
    sent_int$easy_rank <- rev(seq_len(nrow(sent_int)))
    all_difficulty_rows[[length(all_difficulty_rows) + 1]] <- sent_int

    n <- nrow(sent_int)
    n_hard <- min(10, n)
    lines <- c(lines,
      sprintf("\\multicolumn{4}{l}{\\textit{%s -- Hardest}} \\\\", gname))
    for (j in seq_len(n_hard)) {
      r <- sent_int[j, ]
      lines <- c(lines, sprintf(
        " & %d & %s & %.3f \\\\",
        j, esc(trunc_sent(r$sentence)), r$intercept
      ))
    }
    lines <- c(lines, "\\addlinespace[3pt]")

    n_easy <- min(10, n)
    easy <- sent_int[(n - n_easy + 1):n, ]
    easy <- easy[order(-easy$intercept), ]
    lines <- c(lines,
      sprintf("\\multicolumn{4}{l}{\\textit{%s -- Easiest}} \\\\", gname))
    for (j in seq_len(n_easy)) {
      r <- easy[j, ]
      lines <- c(lines, sprintf(
        " & %d & %s & %+.3f \\\\",
        j, esc(trunc_sent(r$sentence)), r$intercept
      ))
    }
    if (i < length(groups)) lines <- c(lines, "\\addlinespace")
  }

  lines <- c(lines, lt$foot)
  writeLines(lines, file.path(output_dir, sprintf("stat_%s_sentence_difficulty.tex", tot)))
  write.csv(do.call(rbind, all_difficulty_rows),
            file.path(output_dir, sprintf("results_%s_sentence_difficulty.csv", tot)),
            row.names = FALSE)
  cat(sprintf("Wrote %s sentence difficulty outputs\n", tot))

  cat(sprintf("\nArchitecture gap by sentence (%s, recurrent vs attention)...\n", tot))

  all_gap_rows <- list()
  cap <- sprintf(paste0("Sentences with the largest recurrent-vs-attention advantage gap ",
                        "(%s sentences). Gap = mean recurrent advantage ",
                        "(SRN, GRU, LSTM) minus mean attention advantage ",
                        "(all Attn variants), averaging over entity condition and seed."), tot)
  hdr <- paste0("\\textbf{Test Group} & \\textbf{Sentence} & ",
                "\\textbf{Recurrent} & \\textbf{Attention} & \\textbf{Gap}")
  lt <- longtable_wrap("@{}llrrr@{}", hdr, cap,
                       sprintf("tab:%s_sentence_arch_gap", tot))
  lines <- lt$head

  for (i in seq_along(groups)) {
    group <- groups[i]
    gname <- group_names[[i]]
    gdf <- data[data$group == group & data$train_or_test == tot, ]

    sent_means <- aggregate(advantage ~ sentence + is_recurrent, data = gdf, FUN = mean)
    rec_means <- sent_means[sent_means$is_recurrent == "recurrent", c("sentence", "advantage")]
    att_means <- sent_means[sent_means$is_recurrent == "attention", c("sentence", "advantage")]
    names(rec_means)[2] <- "rec_mean"
    names(att_means)[2] <- "att_mean"

    gap_df <- merge(rec_means, att_means, by = "sentence")
    gap_df$gap <- gap_df$rec_mean - gap_df$att_mean
    gap_df$group <- group
    gap_df$group_name <- gname
    gap_df <- gap_df[order(-gap_df$gap), ]
    gap_df$gap_rank <- seq_len(nrow(gap_df))
    all_gap_rows[[length(all_gap_rows) + 1]] <- gap_df

    n_show <- min(15, nrow(gap_df))
    top <- gap_df[seq_len(n_show), ]

    first <- TRUE
    for (j in seq_len(n_show)) {
      r <- top[j, ]
      glabel <- if (first) gname else ""
      first <- FALSE
      lines <- c(lines, sprintf(
        "%s & %s & %.3f & %.3f & %+.3f \\\\",
        glabel, esc(trunc_sent(as.character(r$sentence))),
        r$rec_mean, r$att_mean, r$gap
      ))
    }
    if (i < length(groups)) lines <- c(lines, "\\addlinespace")
  }

  lines <- c(lines, lt$foot)
  writeLines(lines, file.path(output_dir, sprintf("stat_%s_sentence_arch_gap.tex", tot)))
  write.csv(do.call(rbind, all_gap_rows),
            file.path(output_dir, sprintf("results_%s_sentence_arch_gap.csv", tot)),
            row.names = FALSE)
  cat(sprintf("Wrote %s sentence architecture-gap outputs\n", tot))
}

write_warning_log(output_dir, "results_sentence_diagnostics_warnings.txt")
cat("\nDone.\n")
