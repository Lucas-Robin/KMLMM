
library(foreach)
library(doRNG)
library(doSNOW)
library(kernlab)
registerDoSNOW(cl <- makeCluster(8, type = "SOCK"))
set.seed(42)

ZIP_train <- read.table("zip_train.dat")
colnames(ZIP_train) <- c("Response", sapply(1:256, function(x) paste("V", x, sep = "_")))

ZIP_test <- read.table("zip_test.dat")
colnames(ZIP_test) <- c("Response", sapply(1:256, function(x) paste("V", x, sep = "_")))

train <- ZIP_train

Y <- train$Response

X <- as.matrix(train[, - 1])

Xscaled <- scale(X, center = TRUE, scale = FALSE)



# gtools::combinations(10, 2, 0:9)
# gtools::combinations(10, 3, 0:9)
# gtools::combinations(10, 4, 0:9)
# gtools::combinations(10, 5, 0:9)

mod <- list()

for(i in 0:9)
{
  mod[[i+1]] <- ksvm(X, Y == i, scaled = FALSE)
}