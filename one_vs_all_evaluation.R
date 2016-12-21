library(kernlab)

source("zip.R")
source("mnist.R")
source("one_vs_all_predictor.R")

c_candidates = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)
k = 5
#c_candidates = c(1, 10, 100)
#k = 3

data = read_zip()
#data = read_mnist()

X_train = data$X_train
X_test = data$X_test
y_train = data$y_train
y_test = data$y_test

n_train = dim(X_train)[1]
n_valid = 1/k * n_train
indices = sample(1:n_train)

mean_accuracies = c()
for (j in 1:length(c_candidates)) {
  C = c_candidates[j]
  print("-----")
  print(paste("C =", C))
  accuracies = c()
  for (i in 1:k) {
    print(paste("fold", i, "/", k))
    # divide training set
    valid_indices = indices[((i-1)*n_valid):(i*n_valid)]
    X = X_train[-valid_indices, ]
    y = y_train[-valid_indices]
    X_valid = X_train[valid_indices, ]
    y_valid = y_train[valid_indices]
    # build models
    models = one_vs_all.models(X, y, C, verbose = FALSE)
    # evaluate models on validation set
    predictions = one_vs_all.predict(models, X_valid)
    labels = one_vs_all.get_classes(predictions)
    accuracy = mean(labels == y_valid)
    accuracies[i] = accuracy
  }
  mean_accuracy = mean(accuracies)
  print(paste("mean accuracy =", mean_accuracy))
  mean_accuracies[j] = mean_accuracy
}

# plot mean accuracy over C
plot(c_candidates, mean_accuracies,
     ylab="mean CV accuracy",
     xlab="C",
     type="l",
     col="red",
     log="x")

# select the C with highest validation accuracy
best = which.max(mean_accuracies)
C = c_candidates[best]
print(paste("best C is", C))
print(paste("with a validation accuracy of", mean_accuracies[best]))

# build models
models = one_vs_all.models(X_train, y_train, C, verbose = FALSE)
# evaluate models on test set
predictions = one_vs_all.predict(models, X_test)
labels = one_vs_all.get_classes(predictions)
accuracy = mean(labels == y_test)
print(paste("test accuracy =", accuracy))