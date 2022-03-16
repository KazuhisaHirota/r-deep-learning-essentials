source("activation_function.R")

library(R6)

Perceptrons <- R6Class(
  "Perceptrons",
  
  private = list(
    n.in = NULL,
    w = NULL
  ),
  
  public = list(
    
    initialize = function(n.in) {
      
      private$n.in <- n.in
      private$w <- numeric(n.in)
    },

    Train = function(x, t, learning.rate) {
      # check if the data is classified correctly
      c <- 0.0
      for (i in 1:private$n.in) {
        c <- c + private$w[i] * x[i] * t
      }
      
      classified <- 0
      if (c > 0.0) {
        classified <- 1
        print(paste("c=", as.character(c),
                    ". the data is classified correctly."))
      } else { # apply steepest descent method if the data is wrongly classified
        for (i in 1:private$n.in) {
          private$w[i] <- private$w[i] + learning.rate * x[i] * t
          print(paste("i=", as.character(i),
                      "w[1]=", as.character(private$w[1]),
                      "w[2]=", as.character(private$w[2])))
        } 
      }
      
      return(classified)
    },

    # calc \sigma(wx)
    Predict = function(x) {
      # wx
      pre.activation <- 0.0
      for (i in 1:private$n.in) {
        pre.activation <- pre.activation + private$w[i] * x[i]
      }
      
      return(Step(pre.activation))
    }
  )
)