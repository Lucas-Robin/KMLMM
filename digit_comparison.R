library(kernlab)

source("zip.R")

zip = read_zip()

X_train = zip$X_train
y_train = zip$y_train
X_test = zip$X_test
y_test = zip$y_test

X_test_centered = scale(X_test, center=colMeans(X_train), scale=F)
X_train_centered = scale(X_train, scale=F)

accuracies = matrix(0, nrow = 10, ncol = 10)

for (i in 0:8) {
  for (j in (i+1):9) {
    train_indices = y_train == i | y_train == j
    test_indices = y_test == i | y_test == j
    model = ksvm(X_train_centered[train_indices, ], as.factor(y_train[train_indices]), scaled=F)
    predictions = predict(model, X_test_centered[test_indices, ])
    accuracy = mean(predictions == y_test[test_indices])
    accuracies[i + 1, j + 1] = accuracy
    accuracies[j + 1, i + 1] = accuracy
    print(paste("accuracy", i, "vs.", j, "=", accuracy))
  }
}