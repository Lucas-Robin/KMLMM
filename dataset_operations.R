
split_dataset <- function(X, y, valid_set_percentage) {
  n = nrow(X)
  n_valid = n * valid_set_percentage
  valid_sample = sample(1:n, n_valid)
  X_train = X[-valid_sample, ]
  y_train = y[-valid_sample]
  X_valid = X[valid_sample, ]
  y_valid = y[valid_sample]
  return(list(
    X_train = X_train,
    y_train = y_train,
    X_valid = X_valid,
    y_valid = y_valid
  ))
}