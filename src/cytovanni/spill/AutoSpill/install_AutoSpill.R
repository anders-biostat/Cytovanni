if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("flowCore")

if (!require("devtools", quietly = TRUE))
    install.packages("devtools")

install_github( "carlosproca/autospill" )
library(autospill)
