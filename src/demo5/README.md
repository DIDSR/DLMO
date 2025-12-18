# Statistical anslysis

Statistical analysis includes two example scripts: 1) sample size determination via a power analysis and 2) statistical analysis for a pivotal study. Pre-installation of the iMRMC application is **NOT** recommended for the use of these scripts.

# [*Sample size determination*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/power_analysis)

This script conducts a power analysis for sample size determination in our paper. To run the script, simply execute `power_analysis_BDG.R`. To use your own pilot data, please replace `pliot_data.csv` with your data following the same format, and update proportion correct by DLMO and its variance in the `power_analysis_BDG.R`.

# [*Pivotal study*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/pivotal_study)

This script conducts a similarity test to investigate whether DLMO performs similarly to human readers within a pre-defined margin of 0.1 proportion correct. To run the script, simply execute `similarity_test.R`. To use it for your own project, please update the `DLMO reading results` section in `similarity_test.R`, and provide reading scores in the `reading_scores` folder following the same format.

