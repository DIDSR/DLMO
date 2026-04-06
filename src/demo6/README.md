# Statistical analysis

Statistical analysis includes two example scripts: 1) sample size determination using a pilot data and 2) statistical analysis for a pivotal study. Pre-installation of the iMRMC application is **NOT** recommended for the use of these scripts.

This demo also includes supporting material in `sample_size_4rm_pilot_study/` and helper R functions in `iMRMC_2AFC/`.

# [*Sample size determination*](http://github.com/DIDSR/DLMO/tree/main/src/demo6/sample_size_4rm_pilot_study)

This script performs sample size determination for a split-plot design, as detailed in Section IV in the [supplementary material](https://arxiv.org/src/2602.22535v1/anc/DLMO_supp.pdf) of our [DLMO paper](https://arxiv.org/abs/2602.22535). To run the script, simply execute `sample_size_est_BDG.R`. To use your own pilot data, replace `pilot_data.csv` with your dataset following the same format, and update the DLMO proportion correct and its variance in `sample_size_est_BDG.R`.

# [*Pivotal study*](https://github.com/DIDSR/DLMO/tree/main/src/demo6/pivotal_study)

This script conducts a similarity test to assess whether our DLMO performs comparably to human readers within a predefined margin of 0.1 in proportion correct. To run the script, simply execute `similarity_test.R`. To use it for your own project, update the ‘DLMO reading results’ section in `similarity_test.R` and provide reading scores in the reading_scores folder following the same format.

