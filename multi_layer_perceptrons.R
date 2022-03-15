source("hidden_layer.R")
source("logistic_regression.R")


MultiLayerPerceptrons <- R6Class(
  "MultiLayerPerceptrons",
  
  private = list(
    n.in = NULL,
    n.hidden = NULL,
    n.out = NULL,
    hidden.layer = NULL,
    logistic.layer = NULL
  ),
  
  public = list(
    initialize = function(n.in, n.hidden, n.out) {
      private$n.in <- n.in
      private$n.hidden <- n.hidden
      private$n.out <- n.out
      private$hidden.layer <- HiddenLayer$new(n.in, n.hidden,
                                              NULL, NULL, "tanh")
      private$logistic.layer <- LogisticRegression$new(n.hidden, n.out)
    },
    
    Train = function(X, T, minibatch.size, learning.rate) {
      # outputs of hidden layer (= inputs of output layer)
      # NOTE ncol = self.n_in in the original code is wrong
      Z = matrix(0, nrow = minibatch.size, ncol = private$n.hidden)
      
      for (n in 1:minibatch.size) {
        # activate input units
        Z[n,] <- private$hidden.layer$Forward(X[n,]) # X: matrix
      }
      
      # forward & backward output layer
      d.Y <- private$logistic.layer$Train(Z, T, minibatch.size, learning.rate)
      
      # backward hidden layer (backpropagate)
      private$hidden.layer$Backward(X, Z, d.Y, 
                                    private$logistic.layer$GetW(),
                                    minibatch.size, learning.rate)
    },
    
    Predict = function(x) {
      z <- private$hidden.layer$Output(x)
      return(private$logistic.layer$Predict(z))
    }
  )
)
