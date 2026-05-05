#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("Specify the single stain data directory!", call.=FALSE)
} else if (length(args)>1) {
  stop("Specify only the single stain data directory, too many arguments!", call.=FALSE)
}

# auto-installing may not work on a cluster anyway, for now just throw error if not installed
library(autospill)
#if(!require(autospill)){
    #library( devtools )
    #install_github( "carlosproca/autospill" )
    #library(autospill)
#}

# Not sure how to best call a script within an R package, for now simply copy their code from https://github.com/carlosproca/autospill/blob/master/inst/batch/


# https://stackoverflow.com/questions/13110076/function-to-concatenate-paths
# system.file("batch", "calculate_compensation_paper.r", package="autospill")
