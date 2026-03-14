# source /projects01/mikem/applications/R-4.4.1/set_env.sh
#
# iMRMC was imported from the source code as follows.

# Set the path to your folder
folder_path <-
  "../iMRMC_2AFC"

# List all R files in the folder
r_files <-
  list.files(path = folder_path,
             pattern = "\\.R$",
             full.names = TRUE)

# Source each file
for (file in r_files) {
  source(file)
}

# -------------------------- Load study ----------------------------------
acc <- '8'
rec_method <- 'unet'

data <- data.frame ()
for (L in 5:8) {
  csv_fname_ho <-
    paste("./reading_scores/", rec_method, "/R1_acc", acc, "_L", L, ".csv", sep = "")
  data_ho <- read.csv(csv_fname_ho,  header = TRUE)
  data_ho$ImageID <- data_ho$ImageID + 40 * (L - 5)
  data <- rbind(data, data_ho)

  csv_fname_ho <-
    paste("./reading_scores/", rec_method, "/R2_acc", acc, "_L", L, ".csv", sep = "")
  data_ho <- read.csv(csv_fname_ho,  header = TRUE)
  data_ho$ImageID <- data_ho$ImageID + 40 * (L - 5)
  data <- rbind(data, data_ho)

  csv_fname_ho <-
    paste("./reading_scores/", rec_method, "/R3_acc", acc, "_L", L, ".csv", sep = "")
  data_ho <- read.csv(csv_fname_ho,  header = TRUE)
  data_ho$ImageID <- data_ho$ImageID + 40 * (L - 5)
  data <- rbind(data, data_ho)

  csv_fname_ho <-
    paste("./reading_scores/", rec_method, "/R4_acc", acc, "_L", L, ".csv", sep = "")
  data_ho <- read.csv(csv_fname_ho,  header = TRUE)
  data_ho$ImageID <- data_ho$ImageID + 40 * (L - 5)
  data <- rbind(data, data_ho)

}

# -------------------------- DLMO reading results ------------------------------
if (acc == 4){
  if (rec_method == 'rsos'){
    # rsos acc4
    AUC_dlmo <- 0.9511991
    var_dlmo <- 2.004043e-06
  }else{
    # unet acc4
    AUC_dlmo <- 0.9789667
    var_dlmo <- 5.287142e-07
  }
}else{
  if (rec_method == 'rsos'){
    # rsos acc8
    AUC_dlmo <- 0.9538379
    var_dlmo <- 1.772338e-06
  }else{
    # unet acc8
    AUC_dlmo <- 0.9620078
    var_dlmo <- 1.19632e-06
  }
}

# -------------------------- Curate study ----------------------------------
dfMRMC_ho <- createIMRMC2AFCdf(
  data,
  keyColumns = list(
    readerID = "User",
    caseID = "ImageID",
    modalityID = "HO",
    score = "Correctness"
  )
)

# ---------------- Estimate variance components ----------------
results <- doIMRMC(dfMRMC_ho)
AUC_HO <- results$Ustat$AUCA
var_HO <- results$Ustat$varAUCA

se_diff <- (var_HO + var_dlmo) ^ .5

diff_AUC <- AUC_dlmo - results$MLEstat$AUCA
diff_AUC_low <- diff_AUC - 2.5 * se_diff
diff_AUC_up <- diff_AUC + 2.5 * se_diff

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