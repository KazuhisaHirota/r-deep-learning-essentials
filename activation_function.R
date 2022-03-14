Step <- function(x) {
  if (x >= 0.0) {
    return(1) 
  } else {
    return(-1)
  }
}

Sigmoid <- function(x) {
  return(1.0 / (1.0 + exp(-x)))
}

DSigmoid <- function(y) {
  return(y * (1.0 - y))
}

Tanh <- function(x) {
  return(tanh(x))
}

DTanh <- function(y) {
  return(1.0 - y * y)
}

Softmax <- function(x) {
  n <- length(x)
  y <- numeric(n)
  
  m <- max(x)
  for (i in 1:n) {
    y[i] <- exp(x[i] - m)
  }
  
  s <- sum(y)
  for (i in 1:n) {
    y[i] <- y[i] / s
  }
  
  return(y)
}