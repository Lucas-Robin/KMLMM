#######
#
#
#######


library(foreach)
library(doRNG)
library(doSNOW)
library(kernlab)
library(caret)
registerDoSNOW(cl <- makeCluster(8, type = "SOCK"))
set.seed(42)



########################################################################################
##################################### loading data #####################################
########################################################################################

perData2sample <- 0.05


########## Loading data 
ZIP_train <- read.table("zip_train.dat")
colnames(ZIP_train) <- c("Response", sapply(1:256, function(x) paste("V", x, sep = "_")))

ZIP_test <- read.table("zip_test.dat")
colnames(ZIP_test) <- c("Response", sapply(1:256, function(x) paste("V", x, sep = "_")))

train <- ZIP_train #[sample(1:nrow(ZIP_train), nrow(ZIP_train) * perData2sample),]

# Y <- cbind(is_0=rep(0, nrow(train)), is_1=rep(0, nrow(train)), is_2=rep(0, nrow(train)),
#            is_3=rep(0, nrow(train)), is_4=rep(0, nrow(train)), is_5=rep(0, nrow(train)),
#            is_6=rep(0, nrow(train)), is_7=rep(0, nrow(train)), is_8=rep(0, nrow(train)),
#            is_9=rep(0, nrow(train)))
# 
# 
# Y[which(train[, "Response"] == 0), "is_0"] <- 1
# Y[which(train[, "Response"] == 1), "is_1"] <- 1
# Y[which(train[, "Response"] == 2), "is_2"] <- 1
# Y[which(train[, "Response"] == 3), "is_3"] <- 1
# Y[which(train[, "Response"] == 4), "is_4"] <- 1
# Y[which(train[, "Response"] == 5), "is_5"] <- 1
# Y[which(train[, "Response"] == 6), "is_6"] <- 1
# Y[which(train[, "Response"] == 7), "is_7"] <- 1
# Y[which(train[, "Response"] == 8), "is_8"] <- 1
# Y[which(train[, "Response"] == 9), "is_9"] <- 1

Y <- train$Response

X <- as.matrix(train[, - 1])

Xscaled <- scale(X, center = TRUE, scale = FALSE)


########################################################################################
######################## First quick (not so much) test of svm #########################
########################################################################################


mod.spoc <- ksvm(X, as.factor(Y), type = "spoc-svc", scaled = FALSE)

prediction.spoc <- predict(mod.spoc, ZIP_test[, -1])
errors.spoc <- which(prediction.spoc != ZIP_test[, "Response"])
prediction.spoc[errors.spoc]
ZIP_test[errors.spoc, "Response"]

mean(prediction.spoc == ZIP_test[, "Response"])


mod.kbb <- ksvm(X, as.factor(Y), type = "kbb-svc", scaled = FALSE)

prediction.kbb <- predict(mod.kbb, ZIP_test[, -1])
errors.kbb <- which(prediction.kbb != ZIP_test[, "Response"])
prediction.kbb[errors.kbb]
ZIP_test[errors.kbb, "Response"]

mean(prediction.kbb == ZIP_test[, "Response"])

########################################################################################
### First I'm just going to compare ksvm cross validation with a custom parallel one ###
########################################################################################

ksvm.crossVal <- function(x, y = NULL, type = NULL,kernel = "rbfdot", 
                          kpar = "automatic", C = 1, nu = 0.2, epsilon = 0.1, 
                          prob.model = FALSE, class.weights = NULL, nfolds = 5, fit = TRUE, 
                          cache = 40, tol = 0.001, shrinking = TRUE, scaled = TRUE)
{
  
  folds <- createFolds(y, k = nfolds)
  
  res <- foreach(i = 1:nfolds) %dorng%
  {
    library(kernlab)
    library(caret)
    # training dataset
    x_train = x[-folds[[i]],]
    y_train = y[-folds[[i]]]
    # testing dataset
    x_test = x[folds[[i]],]
    y_test = y[folds[[i]]]
    
    if(!is.factor(y_train))
    {
      y_train <- as.factor(y_train)
      y_test <- as.factor(y_test)
      
    }
    
    mod.cv <- ksvm(x_train, y = y_train, type = type, kernel = kernel, kpar = kpar, C = C, nu = nu, 
                   epsilon = epsilon, prob.model = prob.model,  class.weights = class.weights, 
                   fit = fit, cache = cache, tol = tol, shrinking = shrinking, scaled = scaled)
    
    prediction <- predict(mod.cv, newdata = x_test)
    c.mat.cv <- table(prediction, y_test)
    
    accuracy <- confusionMatrix(c.mat.cv)$overall[["Accuracy"]]
    
    return(list(mod.cv = mod.cv, c.mat.cv = c.mat.cv, accuracy = accuracy))
  }
  
  accuracy <- sapply(res, function(x) x$accuracy)
  return(list(result = res, accuracy = accuracy, accuracy.mean = mean(accuracy)))
}

system.time(test <- ksvm.crossVal(X, Y, type = "spoc-svc", scaled = FALSE, nfolds = 5))
system.time(test2 <- ksvm(X, as.factor(Y), type = "spoc-svc", scaled = FALSE, cross = 5))


