#!/usr/bin/env Rscript

command_args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", command_args, value = TRUE)
script_path <- if (length(script_arg) > 0) {
  normalizePath(sub("^--file=", "", script_arg[1]))
} else {
  normalizePath(getwd())
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."))

input_root_default <- file.path(repo_root, "statistical_analysis")
output_dir_default <- file.path(input_root_default, "compact_sentence_difficulty_summaries")

args <- commandArgs(trailingOnly = TRUE)
input_root <- if (length(args) >= 1) args[1] else input_root_default
output_dir <- if (length(args) >= 2) args[2] else output_dir_default
top_k <- if (length(args) >= 3) as.integer(args[3]) else 3L

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

group_order <- c("Word", "Sentence", "Complex Event", "Basic Event")

latex_escape <- function(text) {
  replacements <- c(
    "\\" = "\\\\textbackslash{}",
    "&" = "\\\\&",
    "%" = "\\\\%",
    "$" = "\\\\$",
    "#" = "\\\\#",
    "_" = "\\\\_",
    "{" = "\\\\{",
    "}" = "\\\\}"
  )
  out <- text
  for (old in names(replacements)) {
    out <- gsub(old, replacements[[old]], out, fixed = TRUE)
  }
  out
}

truncate_sentence <- function(text, limit = 56L) {
  if (nchar(text) <= limit) {
    return(text)
  }
  paste0(substr(text, 1L, limit - 3L), "...")
}

detect_voice <- function(sentence) {
  if (grepl("\\bis\\b.*\\bby\\b", sentence)) "Passive" else "Active"
}

detect_profile <- function(sentence) {
  has_location <- grepl("bathroom|bedroom|shower|playground|street|outside|inside", sentence)
  has_manner <- grepl(" well\\b| badly\\b|with ease|with difficulty", sentence)
  if (has_location && has_manner) return("Location+Manner")
  if (has_location) return("Location")
  if (has_manner) return("Manner")
  "Bare"
}

detect_family <- function(group_name, sentence) {
  if (group_name == "Word") {
    return("Soccer/Football")
  }
  if (group_name == "Sentence") {
    if (grepl("\\bloses to\\b|\\blost to\\b", sentence)) return("Lose")
    return("Beat")
  }
  if (group_name == "Complex Event") {
    if (grepl("hide-and-seek", sentence)) return("Hide-and-seek")
    if (grepl("chess", sentence)) return("Chess")
    return("Other")
  }
  if (group_name == "Basic Event") {
    if (grepl("ball", sentence)) return("Ball")
    if (grepl("doll", sentence)) return("Doll")
    if (grepl("puzzle|jigsaw", sentence)) return("Puzzle/Jigsaw")
    return("Other")
  }
  "Other"
}

make_type_label <- function(group_name, sentence) {
  paste(
    detect_family(group_name, sentence),
    detect_voice(sentence),
    detect_profile(sentence),
    sep = " | "
  )
}

write_table <- function(path, column_spec, headers, body_rows, caption, label) {
  lines <- c(
    "\\begin{table}[t]",
    "\\centering\\scriptsize",
    sprintf("\\begin{tabular}{%s}", column_spec),
    "\\toprule",
    paste(headers, collapse = " & "),
    "\\\\",
    "\\midrule"
  )
  if (length(body_rows) == 0L) {
    lines <- c(lines, sprintf("\\multicolumn{%d}{c}{No rows.} \\\\", length(headers)))
  } else {
    for (row in body_rows) {
      lines <- c(lines, paste(row, collapse = " & "), "\\\\")
    }
  }
  lines <- c(
    lines,
    "\\bottomrule",
    "\\end{tabular}",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\end{table}"
  )
  writeLines(lines, path)
}

load_rows <- function(input_root) {
  rows <- read.csv(
    file.path(input_root, "results_test_sentence_difficulty.csv"),
    stringsAsFactors = FALSE
  )
  rows$intercept <- as.numeric(rows$intercept)
  rows[, c("group", "group_name", "sentence", "intercept")]
}

build_type_summary <- function(rows, top_k) {
  rows$type_label <- mapply(make_type_label, rows$group_name, rows$sentence)
  summary_rows <- list()

  for (group_name in group_order) {
    gdf <- rows[rows$group_name == group_name, , drop = FALSE]
    if (nrow(gdf) == 0L) {
      next
    }

    type_names <- sort(unique(gdf$type_label))
    agg_rows <- list()
    for (type_name in type_names) {
      tdf <- gdf[gdf$type_label == type_name, , drop = FALSE]
      hardest_idx <- order(tdf$intercept, tdf$sentence)[1L]
      easiest_idx <- order(-tdf$intercept, tdf$sentence)[1L]
      agg_rows[[length(agg_rows) + 1L]] <- data.frame(
        group_name = group_name,
        type_label = type_name,
        n = nrow(tdf),
        mean_intercept = mean(tdf$intercept),
        hardest_example = tdf$sentence[hardest_idx],
        easiest_example = tdf$sentence[easiest_idx],
        stringsAsFactors = FALSE
      )
    }
    agg <- do.call(rbind, agg_rows)

    hardest <- agg[order(agg$mean_intercept, agg$type_label), , drop = FALSE]
    hardest <- hardest[seq_len(min(top_k, nrow(hardest))), , drop = FALSE]
    hardest$side <- "Hardest"
    hardest$rank <- seq_len(nrow(hardest))
    hardest$example <- hardest$hardest_example

    easiest <- agg[order(-agg$mean_intercept, agg$type_label), , drop = FALSE]
    easiest <- easiest[seq_len(min(top_k, nrow(easiest))), , drop = FALSE]
    easiest$side <- "Easiest"
    easiest$rank <- seq_len(nrow(easiest))
    easiest$example <- easiest$easiest_example

    summary_rows[[length(summary_rows) + 1L]] <- rbind(hardest, easiest)
  }

  out <- do.call(rbind, summary_rows)
  out <- out[order(match(out$group_name, group_order), out$side, out$rank), , drop = FALSE]
  rownames(out) <- NULL
  out
}

build_body_rows <- function(summary_df) {
  body_rows <- list()
  for (i in seq_len(nrow(summary_df))) {
    row <- summary_df[i, ]
    body_rows[[i]] <- c(
      row$group_name,
      row$side,
      as.character(row$rank),
      latex_escape(row$type_label),
      sprintf("%.3f", row$mean_intercept),
      as.character(row$n),
      latex_escape(truncate_sentence(row$example))
    )
  }
  body_rows
}

input_rows <- load_rows(input_root)
summary_df <- build_type_summary(input_rows, top_k)

write_table(
  file.path(output_dir, "stat_test_sentence_difficulty_types.tex"),
  "@{}lllrlrl@{}",
  c(
    "\\textbf{Group}",
    "\\textbf{Side}",
    "\\textbf{Rank}",
    "\\textbf{Type}",
    "\\textbf{Mean}",
    "\\textbf{N}",
    "\\textbf{Example}"
  ),
  build_body_rows(summary_df),
  paste0(
    "Test-side sentence-difficulty summary by sentence type. ",
    "Types combine predicate family, voice, and modifier profile; rows report the top ",
    top_k,
    " hardest and easiest type averages per group."
  ),
  "tab:test_sentence_difficulty_types"
)

cat(sprintf(
  "Wrote compact test sentence-difficulty type summaries to %s\n",
  output_dir
))
