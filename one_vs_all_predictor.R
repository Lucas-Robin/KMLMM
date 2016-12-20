library(kernlab)

source("zip.R")

#source("ksvm_mnist.R") # TODO replace this

data = read_zip()

X_train = data$X_train
X_test = data$X_test
y_train = data$y_train
y_test = data$y_test

#tmp = ksvm.crossVal(X_train, y_train, nfolds = 10, type="kbb-svc")

models = list()
for (i in 0:9) {
  print(paste("computing model", i, "..."))
  model = ksvm(as.factor(y_train == i) ~ as.matrix(X_train))
  models[i + 1] = model
}

pred <- function(data) {
  n = dim(data)[1]
  p = matrix(NA, nrow = n, ncol = 10)
  for (i in 1:10) {
    p[, i] = predict(models[[i]], newdata=as.matrix(data))
  }
  return(p - 1)
}

get_classes <- function(p) {
  return(apply(p, 1, which.max) - 1)
}

predictions = pred(X_test)
labels = get_classes(predictions)
accuracy = mean(labels == y_test)
print(paste("accuracy =", accuracy))

num_preds = apply(predictions, 1, sum)
n_zeros = sum(num_preds == 0)
print(paste("#(no prediction) =", zeros))
n_conflicts = sum(num_preds > 1)
print(paste("#(more than 1 prediction) = ", conflicts))