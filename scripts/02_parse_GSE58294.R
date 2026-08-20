suppressPackageStartupMessages({library(GEOquery); library(data.table)})
gse <- "GSE58294"
outdir <- "D:/TT paper/0811Temporal Pathway/data/raw/GSE58294"
f <- file.path(outdir, paste0(gse, "_series_matrix.txt.gz"))
gset <- getGEO(filename = f, getGPL = FALSE)
eset <- gset
expr <- exprs(eset)
pheno <- pData(eset)
fwrite(as.data.frame(expr), file.path(outdir, "expr.csv"), row.names = TRUE)
fwrite(as.data.frame(pheno), file.path(outdir, "pheno.csv"), row.names = TRUE)
cat("OK samples x probes:", ncol(expr), "x", nrow(expr), "\n")
cat("pheno columns:", ncol(pheno), "\n")