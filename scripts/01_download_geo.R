suppressPackageStartupMessages({
  library(GEOquery)
  library(data.table)
})
options(timeout = 1200)
options(download.file.method.GEOquery = "auto")

download_one <- function(gse, outdir) {
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  cat("==== ", gse, " ====\n", sep = "")
  ok <- FALSE
  for (attempt in 1:4) {
    res <- tryCatch(getGEO(gse, GSEMatrix = TRUE, AnnotGPL = TRUE, destdir = outdir),
                    error = function(e) { cat("  attempt", attempt, "fail:", conditionMessage(e), "\n"); NULL })
    if (!is.null(res) && length(res) > 0) {
      ok <- TRUE
      eset <- res[[1]]
      expr <- exprs(eset)
      pheno <- pData(eset)
      fwrite(as.data.frame(expr), file.path(outdir, "expr.csv"), row.names = TRUE)
      fwrite(as.data.frame(pheno), file.path(outdir, "pheno.csv"), row.names = TRUE)
      cat("  SAVED:", gse, ncol(expr), "samples x", nrow(expr), "probes\n")
      break
    }
    Sys.sleep(10)
  }
  if (!ok) cat("  FAILED:", gse, "\n")
  sf <- tryCatch(getGEOSuppFiles(gse, baseDir = outdir, makeDirectory = FALSE),
                 error = function(e) NULL)
  if (!is.null(sf)) cat("  supp files:", nrow(sf), "\n") else cat("  no supp files\n")
}

args <- commandArgs(trailingOnly = TRUE)
gses <- if (length(args) >= 1) strsplit(args[1], ",")[[1]] else c("GSE37587","GSE58294","GSE16561","GSE22255","GSE202709")
rawdir <- if (length(args) >= 2) args[2] else "D:/TT paper/0811Temporal Pathway/data/raw"
for (g in gses) download_one(g, file.path(rawdir, g))
cat("ALL DONE\n")