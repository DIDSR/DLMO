# source R-4.4.1/set_env.sh
#
#
# This R script implements a split-plot design to 
# estimate the number of cases each reader must 
# review when comparing two methods.

# Refer to the PDF file sizing_2AFC.pdf in the attached 
# folder for more information about the split-plot design.

# Use this script to adjust the number of cases (N_C_f) 
# and the number of readers (N_R_f) until the output indicates a
# “Good design.” Once achieved, these N_C_f and N_R_f values can be
# used to conduct the final pivotal study.
# 
# --------------------------------------------
# INPUT guess on the no. of readers
# & no. of cases for pivotal study/
# Change these numbers until you 
# get output as “Good design.”
# ----------------------------------------------
N_R_f <- 2         # no. of readers in pivotal study
N_C_f <- 400       # no. of total cases in pivotal study
bfc   <- 4 

# ------------------------------------------------------------------------
# downloading iMRMC cran is not recommended to run this script.
# Rather import iMRMC source codes uploaded in this repository
# as follows: 
folder_path <- "../iMRMC_2AFC"

# List all R files in the folder
r_files <-
  list.files(path = folder_path,
             pattern = "\\.R$",
             full.names = TRUE)

# Source each file
for (file in r_files) {
  source(file)
}

#----------------------------------------------------------------
# PILOT STUDY
#----------------------------------------------------------------

# Load pilot study data
csv_fname_ho <- "pilot_data.csv"

cat("reading csv file:", csv_fname_ho, "\n")
data_ho <- read.csv(csv_fname_ho,  header = TRUE)
dfMRMC_ho <- createIMRMC2AFCdf(
  data_ho,
  keyColumns = list(
    readerID = "User",
    caseID = "ImageID",
    modalityID = "HO",
    score = "Correctness"
  )
)
results <- doIMRMC(dfMRMC_ho)
margin <- 0.1

# -------------------------- DLMO reading results ------------------------------
AUC_dlmo = 0.879294187
var_dlmo = ((0.879294187 - 0.857929277) / 1.96) ^ 2

# -------------------------- Extract Moments -------------------------------
bdg_comp_df <- results[["varDecomp"]][["BDG"]][["MLE"]][["comp"]]
M1 <- as.numeric(bdg_comp_df[1, "M1"])
M4 <- as.numeric(bdg_comp_df[1, "M4"])
M5 <- as.numeric(bdg_comp_df[1, "M5"])
M8 <- as.numeric(bdg_comp_df[1, "M8"])

# ---- Define specific parameters ----
N_R <- results$full$summaryMRMC$nR    # Number of readers (N_gamma)
N_C <- results$full$summaryMRMC$nC.pos   # Number of cases (N_g)

# ---- Compute BDG coefficients for fully crossed ----
c1fc <- 1 / (N_R * N_C)
c4fc <- (N_C - 1) / (N_R * N_C)
c5fc <- (N_R - 1) / (N_R * N_C)
c8fc <- ((N_C - 1) * (N_R - 1) - (N_R * N_C)) / (N_R * N_C)

# Define functions for standard error for fully crossed
calc_se_mrmc <- function(c1, c4, c5, c8, M1, M4, M5, M8) {
  # Compute the square root of the weighted sum of the moments.
  var_mrmc <- c1 * M1 + c4 * M4 + c5 * M5 + c8 * M8
  se_mrmc <- sqrt(var_mrmc)
  return(se_mrmc)
}

se_BDG <- calc_se_mrmc(c1fc, c4fc, c5fc, c8fc, M1, M4, M5, M8)

var_c <-
  results$varDecomp$BCK$MLE$comp$D * results$varDecomp$BCK$MLE$coeff$D
var_r <-
  results$varDecomp$BCK$MLE$comp$R * results$varDecomp$BCK$MLE$coeff$R
var_cr <-
  results$varDecomp$BCK$MLE$comp$DR * results$varDecomp$BCK$MLE$coeff$DR

# ------------------------ Difference in AUCs and its CI -----------------------
diff_AUC <- 0.88 - 0.93 #results$MLEstat$AUCA
se_diff <- (se_BDG ^ 2 + var_dlmo) ^ .5
diff_AUC_low <- diff_AUC - 2.5 * se_diff
diff_AUC_up <- diff_AUC + 2.5 * se_diff

cat("\n")
cat("Pilot study results:", "\n")
cat(paste("se_diff:", se_diff, "\n"))
cat(paste(
  "Difference in AUCs: ",
  diff_AUC,
  " [",
  diff_AUC_low,
  ", ",
  diff_AUC_up,
  "]",
  "\n"
))

# ----------------------------------------------------------------------------
# Sample size output for pivotal study
# ----------------------------------------------------------------------------
z_alpha_2_sided <- qnorm(1 - 0.05/(2*bfc))
c1sp <- 1 / N_C_f
c4sp <- (N_C_f - N_R_f) / (N_R_f * N_C_f)
c5sp <- 0
c8sp <- -1 / N_R_f

se_BDG <- calc_se_mrmc(c1sp, c4sp, c5sp, c8sp, M1, M4, M5, M8)

# new se_diff value changes with changes in no. of final readers
# this does not align with pdf doc. 
new_se_diff <- (se_BDG ^ 2 + var_dlmo) ^ .5

diff_AUC_low <- diff_AUC - z_alpha_2_sided * new_se_diff
diff_AUC_up <- diff_AUC + z_alpha_2_sided * new_se_diff

cat("", "\n")
cat("New design:", "\n")

cat("Split-plot:", "\n")
cat(paste("Number of trials: ", N_C_f, "\n"))
cat(paste("Number of readers: ", N_R_f, "\n"))
cat(paste("Number of trials per reader: ", N_C_f / N_R_f, "\n"))

cat(paste(
  "Difference in AUCs: ",
  diff_AUC,
  " [",
  diff_AUC_low,
  ", ",
  diff_AUC_up,
  "]",
  "\n"
))

if (diff_AUC_low < -margin || diff_AUC_up > margin) {
  cat("Not a good design.", "\n")
} else{
  cat("Good design.", "\n")
}
