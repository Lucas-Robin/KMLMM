##########################################################################
# Functions and some scripts for ksvm multi-class parameter optimisation #
# Lucas Robin - Matthias Hertel                                          #
##########################################################################


library(foreach)
library(doRNG)
library(doSNOW)
library(kernlab)
library(caret)

source("cifar10.R")
source("mnist.R")
source("zip.R")

perData2sample <- 0.05

registerDoSNOW(cl <- makeCluster(parallel::detectCores() -1, type = "SOCK"))
set.seed(42)



ksvm_param_optimisation <- function(X, Y, type = c("spoc-svc", "kbb-svc"), 
                                    kernel = c("rbfdot", "polydot", "vanilladot", "tanhdot", 
                                               "laplacedot", "besseldot", "anovadot", "splinedot"),
                                    kpar = "automatic", C = c(0.1, 0.5, 0.9))
{
  niter <- length(type) * length(kernel) * length(C)
  foreach(i = 0:(niter -1), .export = c("ksvm.crossVal", "createFolds")) %dorng%
  {
    print(paste("test", i, "type =", type[i %% length(type)], "kernel =", 
                kernel[i %% length(kernel)], "C =", C[i %% length(C)]))
    return(ksvm.crossVal(X, Y, type = type[(i %% length(type)) +1], kernel = kernel[(i %% length(kernel)) +1], 
                         kpar = kpar, C = C[(i %% length(C)) +1]))
  
  }
}

mnist.data <- read_mnist()

indiv <- sample(1:nrow(mnist.data$X_train), nrow(mnist.data$X_train) * perData2sample)
X <- mnist.data$X_train[indiv, ]
Y <- mnist.data$y_train[indiv]

system.time(test <- ksvm(X, as.factor(Y), type = "spoc-svc", 
                                  scaled = FALSE))

system.time(test <- ksvm.crossVal(X, as.factor(Y), type = "spoc-svc", 
                                  scaled = FALSE, nfolds = 10))

system.time(param_opt.res <- ksvm_param_optimisation(X, as.factor(Y), C = 10^seq(from = -3, to = 3), kernel = c("rbfdot", "polydot")))

