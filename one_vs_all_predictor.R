library(kernlab)

one_vs_all.models <- function(X, y, C = 1, verbose = TRUE) {
  classes = sort(unique(y))
  models = list()
  for (c in classes) {
    if (verbose) {
      print(paste("computing model", c, "..."))
    }
    model = ksvm(y == c ~ as.matrix(X), C = C)
    models[c + 1] = model
  }
  return(models)
}

one_vs_all.predict <- function(models, data) {
  n = dim(data)[1]
  p = matrix(NA, nrow = n, ncol = 10)
  for (i in 1:10) {
    p[, i] = predict(models[[i]], newdata=as.matrix(data))
  }
  return(p - 1)
}

one_vs_all.get_classes <- function(predictions) {
  return(apply(predictions, 1, which.max) - 1)
}