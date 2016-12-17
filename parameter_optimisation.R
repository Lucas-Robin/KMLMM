


ksvm_param_optimisation <- function(X, Y, type = c("spoc-svc", "kbb-svc"), 
                                    kernel = c("rbfdot", "polydot", "vanilladot", "tanhdot", 
                                               "laplacedot", "besseldot", "anovadot", "splinedot"),
                                    kpar = "automatic", C = c(0.1, 0.5, 0.9))
{
  niter <- length(type) + length(kernel) + length(C)
  foreach(i = 1:niter, .export = c("ksvm.crossVal", "createFolds")) %dorng%
  {
    print(paste("test", i, "type =", type[i %% length(type) +1], "kernel =", 
                kernel[i %% length(kernel) +1], "C =", C[i %% length(C) +1]))
    return(ksvm.crossVal(X, Y, type = type[i %% length(type) +1], 
                         kernel = kernel[i %% length(kernel) +1], kpar = kpar, C = C[i %% length(C) +1]))
  
  }
}



