source("zip.R")
source("mnist.R")
source("cifar10.R")

dataset = 2

if (dataset == 0) {
  data = read_zip()
} else if (dataset == 1) {
  data = read_mnist()
} else if (dataset == 2) {
  data = read_cifar10()
}

X = data$X_train
y = data$y_train

if (dataset == 0) {
  X = (X + 1) * (256 / 2)
}

par(mfrow=c(2,5))

for (i in 0:9) {
  pxls = X[y == i, ][1, ]
  if (dataset < 2) {
    show_digit(pxls)
  } else {
    plot_image(pxls)
  }
}

par(mfrow=c(1,1))

if (dataset == 2) {
  i = 9
  pxls = X[y == i, ][1, ]
  plot_image(pxls)
}