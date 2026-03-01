library(iMRMC)
setwd("C:/backups/DLIR/SOM/cad_dlmo/sizing")

theta_1=0.88
v_1=0.0001188201

csv_fname_ho <- "pilot_study_curated.csv"
data_ho <- read.csv(csv_fname_ho, header = TRUE)
# Analyze the pilot data use this function.
# Note: The pilot data was initially binary but has been curated so that can be used in iMRMC.
# More info: https://github.com/DIDSR/iMRMC/wiki/MRMC-analysis-of-binary-data
results <- doIMRMC(data_ho)

V_0 <- results$MLEstat$varAUCA
cat("Variance of PC achieved by human readers in the pilot study: ",V_0,sep = "")

bdg_comp <- results[["varDecomp"]][["BDG"]][["MLE"]][["comp"]]
M1 <- as.numeric(bdg_comp[1, "M1"])
M4 <- as.numeric(bdg_comp[1, "M4"])
M5 <- as.numeric(bdg_comp[1, "M5"])
M8 <- as.numeric(bdg_comp[1, "M8"])

theta_0 <- results$MLEstat$AUCA
cat("PC achieved by human readers in the pilot study: ",theta_0,sep = "")
##

N_C_start <- 300
N_C_end <- 450
# Set N_R range
N_R_start <- 4
N_R_end <- 10
# Create an empty figure
op <- par(mar=c(5, 6, 4, 2) + 0.1)
plot(1, type = "n",
     xlim = c(N_C_start, N_C_end),
     ylim = c(0.0145, 0.017),
     xlab = "Number of Cases",
     ylab = "Standard Error of PC \n(from the maximum likelihood estimate MRMC Analysis)",
     main = "Standard Error vs Number of Cases")
par(op)
# Define line colors
library(RColorBrewer)
colors <- brewer.pal(50, "Set2")

# First loop over reader numbers
R_index <- 1
for (N_R in seq(N_R_start, N_R_end, by = 2)){
  se_BDG_array_fc_r <-rep(0, 1) # array using replicate command
  # With N_R readers, loop over case numbers
  index <- 1
  for (N_C in N_C_start:N_C_end) {
    # ---- Compute BDG coefficients for fully crossed ----
    c1fc <- 1 / (N_R * N_C)
    c4fc <- (N_C - 1) / (N_R * N_C)
    c5fc <- (N_R - 1) / (N_R * N_C)
    c8fc <- ((N_C - 1) * (N_R - 1) - (N_R * N_C)) / (N_R * N_C)
    var_ho_fc <- c1fc * M1 + c4fc * M4 + c5fc * M5 + c8fc * M8
    se_BDG_array_fc_r[index] <- (var_ho_fc + v_1)^.5
    index <- index + 1
  }
  # Plot standard error with N_R readers and N_C cases
  lines(
    N_C_start:N_C_end,
    se_BDG_array_fc_r,
    col = colors[R_index],
    type = "l",
  )
  R_index <- R_index + 1
}

lines(N_C_start:N_C_end,
      rep(0.0155, index - 1) ,
      lty = 2,
      col = "black")
legend(
  "topright",
  legend = c("N_R = 4","N_R = 6","N_R = 8","N_R = 10", "Target SE"),
  col = c(colors[1:4], "black"),
  lty = c(1,1,1,1,2)
)