source("hidden_layer.R")
source("logistic_regression.R")


MultiLayerPerceptrons <- R6Class(
  "MultiLayerPerceptrons",
  
  private = list(
    n.in = NULL,
    n.hidden = NULL,
    n.out = NULL,
    hidden.layer = NULL,
    # TODO
    logistic.layer.n.in = NULL,
    logistic.layer.n.out = NULL,
    logistic.layer.W = NULL,
    logistic.layer.b = NULL
  ),
  
  public = list(
    initialize = function(n.in, n.hidden, n.out) {
      private$n.in <- n.in
      private$n.hidden <- n.hidden
      private$n.out <- n.out
      private$hidden.layer <- HiddenLayer$new(n.in, n.hidden,
                                              NULL, NULL, "tanh")
      # TODO
      private$logistic.layer.n.in <- n.hidden
      private$logistic.layer.n.out <- n.out
      private$logistic.layer.W <- matrix(0, nrow=n.out, ncol=n.hidden)
      private$logistic.layer.b <- numeric(n.out)
    },
    
    Train = function(X, T, minibatch.size, learning.rate) {
      # outputs of hidden layer (= inputs of output layer)
      Z = matrix(0, nrow = minibatch.size, ncol = private$n.hidden) # NOTE ncol = self.n_in in the original code is wrong
      
      for (n in 1:minibatch.size) {
        # activate input units
        Z[n,] <- private$hidden.layer$Forward(X[n,]) # X: matrix
      }
      
      # forward & backward output layer
      result <- TrainLogisticRegression(private$logistic.layer.n.in,
                                        private$logistic.layer.n.out,
                                        private$logistic.layer.W,
                                        private$logistic.layer.b,
                                        Z, T, minibatch.size, learning.rate)
      d.Y <- result$d.Y
      private$logistic.layer.W <- result$W
      private$logistic.layer.b <- result$b
      
      # backward hidden layer (backpropagate)
      private$hidden.layer$Backward(X, Z, d.Y, 
                                    private$logistic.layer.W,
                                    minibatch.size, learning.rate)
    },
    
    Predict = function(x) {
      z <- private$hidden.layer$Output(x)
      return(PredictLogisticRegression(private$logistic.layer.n.in,
                                       private$logistic.layer.n.out,
                                       private$logistic.layer.W,
                                       private$logistic.layer.b,
                                       z))
    }
  )
)
