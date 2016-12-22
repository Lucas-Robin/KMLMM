# code from https://gist.github.com/brendano/39760 slightly adapted
read_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=n*nrow*ncol,size=1,signed=F)
    x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    return(x)
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    return(y)
  }
  
  X_train = load_image_file('mnist/train-images.idx3-ubyte')
  X_test = load_image_file('mnist/t10k-images.idx3-ubyte')
  
  y_train = load_label_file('mnist/train-labels.idx1-ubyte')
  y_test = load_label_file('mnist/t10k-labels.idx1-ubyte')  
  
  return(list(
    X_train = X_train,
    X_test = X_test,
    y_train = y_train,
    y_test = y_test
  ))
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  n = length(arr784)
  d = sqrt(n)
  image(matrix(as.matrix(arr784), nrow=d)[,d:1], col=col, ...)
}