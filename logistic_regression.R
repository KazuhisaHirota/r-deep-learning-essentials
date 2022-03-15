source("activation_function.R")

library(R6)

LogisticRegression <- R6Class(
  "LogisticRegression",
  
  private = list(
    n.in = NULL,
    n.out = NULL,
    W = NULL,
    b = NULL
  ),
  
  public = list(
    
    initialize = function(n.in, n.out) {
      
      private$n.in <- n.in
      private$n.out <- n.out
      
      private$W <- matrix(0, nrow = n.out, ncol = n.in)
      private$b <- numeric(n.out)
    },
    
    GetW = function() return(private$W),
    Getb = function() return(private$b),
    
    # calc \sigma(W x + b)
    # size = W: (n.out, n.in), x: n.in, b: n.out
    Output = function(x) {
      
      pre.activation = numeric(private$n.out)
      for (j in 1:private$n.out) {
        for (i in 1:private$n.in) {
          pre.activation[j] <- pre.activation[j] + private$W[j, i] * x[i]
        }
        pre.activation[j] <- pre.activation[j] + private$b[j]
      }
      
      return(Softmax(pre.activation))
    },

    Train = function(X, T, minibatch.size, learning.rate) {
      
      grad.W <- matrix(0, nrow=private$n.out, ncol=private$n.in)
      grad.b <- numeric(private$n.out)
      
      d.Y <- matrix(0, nrow=minibatch.size, ncol=private$n.out)
      
      # train with SGD
      
      # calc gradient of W, b
      for (k in 1:minibatch.size) {
        predicted.Y <- self$Output(X[k,])
        
        for (j in 1:private$n.out) {
          d.Y[k, j] <- predicted.Y[j] - T[k, j]
        
          for (i in 1:private$n.in) {
            grad.W[j, i] <- grad.W[j, i] + d.Y[k, j] * X[k, i]
          }
          grad.b[j] <- grad.b[j] + d.Y[k, j]
        }
      }
      
      # update params
      for (j in 1:private$n.out) {
        for (i in 1:private$n.in) {
          # Note a position of starting a new line.
          # If operator is placed at the beginning of a line, the new line is ignored.
          private$W[j, i] <- private$W[j, i] -
                             learning.rate * grad.W[j, i] / minibatch.size
        }
        # Note a position of starting a new line
        # If operator is placed at the beginning of a line, the new line is ignored.
        private$b[j] <- private$b[j] -
                        learning.rate * grad.b[j] / minibatch.size
      }
      
      return(d.Y)
    },

    Predict = function(x) {
      y <- self$Output(x) # probability vector
      t <- numeric(private$n.out) # label casted to 0 or 1
      
      argmax = which.max(y)
      for (i in 1:private$n.out) {
        t[i] <- ifelse(i == argmax, 1, 0) # cast to 0 or 1
      }
      
      return(t)
    }
  )
)




