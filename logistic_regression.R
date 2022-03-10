source("activation_function.R")

# calc \sigma(W x + b)
# size = W: (n.out, n.in), x: n.in, b: n.out
OutputLogisticRegression <- function(n.in, n.out, W, b, x) {
  pre.activation = numeric(n.out)
  for (j in 1:n.out) {
    for (i in 1:n.in) {
      pre.activation[j] <- pre.activation[j] + W[j, i] * x[i]
    }
    pre.activation[j] <- pre.activation[j] + b[j]
  }
  
  return(Softmax(pre.activation))
}

TrainLogisticRegression <- function(n.in, n.out, W, b, X, Tmtx,
                                    minibatch.size, learning.rate) {
  grad.W <- matrix(0, nrow=n.out, ncol=n.in)
  grad.b <- numeric(n.out)
  
  d.Y <- matrix(0, nrow=minibatch.size, ncol=n.out)
  
  # train with SGD
  
  # calc gradient of W, b
  for (k in 1:minibatch.size) {
    predicted.Y <- OutputLogisticRegression(n.in, n.out, W, b, X[k,])
    
    for (j in 1:n.out) {
      d.Y[k, j] <- predicted.Y[j] - Tmtx[k, j]
    
      for (i in 1:n.in) {
        grad.W[j, i] <- grad.W[j, i] + d.Y[k, j] * X[k, i]
      }
      grad.b[j] <- grad.b[j] + d.Y[k, j]
    }
  }
  
  # update params
  for (j in 1:n.out) {
    for (i in 1:n.in) {
      W[j, i] <- W[j, i] - learning.rate * grad.W[j, i] / minibatch.size
    }
    b[j] <- b[j] - learning.rate * grad.b[j] / minibatch.size
  }
  
  return(list(d.Y=d.Y, W=W, b=b))
}

PredictLogisticRegression <- function(n.in, n.out, W, b, x) {
  y <- OutputLogisticRegression(n.in, n.out, W, b, x) # probability vector
  t <- numeric(n.out) # label casted to 0 or 1
  
  argmax = which.max(y)
  for (i in 1:n.out) {
    t[i] <- ifelse(i == argmax, 1, 0) # cast to 0 or 1
  }
  
  return(t)
}




