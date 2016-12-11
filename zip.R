
read_zip <- function() {
  train = read.table("zip_train.dat")
  test = read.table("zip_test.dat")
  ncols = ncol(train)
  y_train = train[, 1]
  X_train = train[, 2:ncols]
  y_test = test[, 1]
  X_test = test[, 2:ncols]
  return(list(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test
  ))
}