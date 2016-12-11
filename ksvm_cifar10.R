library(kernlab)

source("cifar10.R")

# read the dataset
print("reading and centering dataset...")
cifar10 = read_cifar10()

n_training = 3000
n_test = 1000

X_train = cifar10$X_train[1:n_training, ]
y_train = cifar10$y_train[1:n_training]
X_test = cifar10$X_test[1:n_test, ]
y_test = cifar10$y_test[1:n_test]

remove(cifar10)

# center the X data
X_test_centered = scale(X_test, center=colMeans(X_train), scale=F)
X_train_centered = scale(X_train, scale=F)

# build a kernel svm model
print("training svm...")
multiclass_strategy = "spoc-svc"
#multiclass_strategy = "kbb-svc"
model = ksvm(X_train_centered, as.factor(y_train), scaled=F, type=multiclass_strategy)

# evaluate
print("evaluating...")
predictions = predict(model, X_test_centered)
accuracy = mean(predictions == y_test)
print(paste("accuracy with", multiclass_strategy, "on", n_training, "training samples =", accuracy))
