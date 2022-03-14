library(R6)

source("activation_function.R")

HiddenLayer <- R6Class(
  "HiddenLayer",
  
  private = list(
    n.in = NULL,
    n.out = NULL,
    W = NULL,
    b = NULL,
    activation = NULL,
    dactivation = NULL
  ),
  
  public = list(
    
    initialize = function(n.in, n.out, W, b, activation.name) {
      
      if (is.null(W)) {
        W <- matrix(0, nrow = n.out, ncol = n.in)
        limit <- 1. / n.in
        
        for (j in 1:n.out) {
          for (i in 1:n.in) {
            # initialize W with uniform distribution
            W[j, i] <- runif(1, -limit, limit)
          }
        }
      }
      
      if (is.null(b)) b <- numeric(n.out)
      
      private$n.in <- n.in
      private$n.out <- n.out
      private$W <- W
      private$b <- b
      
      if (activation.name == "sigmoid" || is.null(activation.name)) {
        private$activation <- Sigmoid
        private$dactivation <- DSigmoid
      } else if (activation.name == "tanh") {
        private$activation <- Tanh
        private$dactivation <- DTanh
      } else {
        print("Error: this activation function is not supported")
        # TODO
      }
    },
    
    Output = function(x) {
      
      y <- numeric(private$n.out)
      
      for (j in 1:private$n.out) {
        # calc \sigma(W x + b)
        pre.activation <- 0.
        for (i in 1:private$n.in) {
          pre.activation <- pre.activation + private$W[j, i] * x[i]
        }
        pre.activation <- pre.activation + private$b[j]
        
        y[j] <- private$activation(pre.activation)
      }
      
      return(y)
    },
    
    Forward = function(x) {
      return(self$Output(x))
    },
    
    Backward = function(X, Z, d.Y, W.prev, minibatch.size, learning.rate) {
      
      d.Z <- matrix(0, nrow = minibatch.size, ncol = private$n.out) # backpropagation error
      
      grad.W <- matrix(0, nrow = private$n.out, ncol = private$n.in)
      grad.b <- numeric(private$n.out)
      
      # train with SGD
      # calculate backpropagation error to get gradient of W, b
      for (n in 1:minibatch.size) {
        for (j in 1:private$n.out) {
          for (k in 1:dim(d.Y)[2]) { # d.Y column size
            d.Z[n, j] <- d.Z[n, j] + W.prev[k, j] * d.Y[n, k]
          }
          d.Z[n, j] <- d.Z[n, j] * private$dactivation(Z[n, j])
          
          for (i in 1:private$n.in) {
            grad.W[j, i] <- grad.W[j, i] + d.Z[n, j] * X[n, i]
          }
          
          grad.b[j] <- grad.b[j] + d.Z[n, j]
        }
      }
      
      # update params
      for (j in 1:private$n.out) {
        for (i in 1:private$n.in) {
          private$W[j, i] <- private$W[j, i]
                             - learning.rate * grad.W[j, i] / minibatch.size
        }
        private$b[j] <- private$b[j]
                        - learning.rate * grad.b[j] / minibatch.size
      }
      
      return(d.Z)
    }
  )
)
