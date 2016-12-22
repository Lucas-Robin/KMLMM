library(kernlab)

source("zip.R")
source("dataset_operations.R")

# load training set
data = read_zip()
X = data$X_train
y = data$y_train

# divide training set into new (smaller) training set and validation set
valid_set_percentage = 1 / 3
splitted_set = split_dataset(X, y, valid_set_percentage)
X_train = splitted_set$X_train
y_train = splitted_set$y_train
X_valid = splitted_set$X_valid
y_valid = splitted_set$y_valid

# center the X-matrices according to the training set
X_valid_centered = scale(X_valid, center=colMeans(X_train), scale=F)
X_train_centered = scale(X_train, scale=F)

# for each pair of digits: build a ksvm and evaluate it on the validation set
accuracies = matrix(NA, nrow = 10, ncol = 10)

for (i in 0:8) {
  for (j in (i+1):9) {
    train_indices = y_train == i | y_train == j
    valid_indices = y_valid == i | y_valid == j
    model = ksvm(X_train_centered[train_indices, ], as.factor(y_train[train_indices]), scaled=F)
    predictions = predict(model, X_valid_centered[valid_indices, ])
    accuracy = mean(predictions == y_valid[valid_indices])
    accuracies[i + 1, j + 1] = accuracy
    accuracies[j + 1, i + 1] = accuracy
    print(paste("accuracy", i, "vs.", j, "=", accuracy))
  }
}

# For each digit, find the one against which we get the best accuracy
couples <- sapply(1:10, function(i, acc) {return(which.max(acc[i, ]) -1)}, accuracies)
couples[order(couples)]

# For each digit, find the one against which we get the worst accuracy
couples <- sapply(1:10, function(i, acc) {return(which.min(acc[i, ]) -1)}, accuracies)
couples[order(couples)]
