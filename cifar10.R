library(grid)

labels = read.table("cifar-10-batches-bin/batches.meta.txt")

columns = 1 + 3 * 32 * 32

read_batch <- function(batch_index) {
  batch = matrix(data = NA, nrow = 10000, ncol = columns)
  if (batch_index == 0) {
    f <- file("cifar-10-batches-bin/test_batch.bin", "rb")
  } else {
    f <- file(paste("cifar-10-batches-bin/data_batch_", batch_index, ".bin", sep=""), "rb")
  }
  for (i in 1:10000) {
    data = as.integer(readBin(f, raw(), size=1, n=columns, endian="big"))
    batch[i, ] = data
  }
  close(f)
  return(batch)
}

read_cifar10 <- function() {
  data = matrix(data = NA, nrow = 50000, ncol = columns)
  for (i in 1:5) {
    batch = read_batch(i)
    data[((i - 1) * 10000 + 1):(i * 10000), ] = batch
  }
  y_train = data[, 1]
  X_train = data[, 2:columns]
  test = read_batch(0)
  y_test = test[, 1]
  X_test = test[, 2:columns]
  return(list(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test
  ))
}

plot_image <- function(pixels) {
  r <- matrix(pixels[1:1024], ncol=32, byrow = TRUE)
  g <- matrix(pixels[1025:2048], ncol=32, byrow = TRUE)
  b <- matrix(pixels[2049:3072], ncol=32, byrow = TRUE)
  pixels <- rgb(r, g, b, maxColorValue = 255)
  dim(pixels) <- c(32, 32)
  grid.raster(pixels, interpolate=FALSE)
}