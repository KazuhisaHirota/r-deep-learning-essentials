source("activation_function.R")

TrainPerceptrons <- function(n.in, w, x, t, learning.rate) {
  # check if the data is classified correctly
  c <- 0.0
  for (i in 1:n.in) {
    c <- c + w[i] * x[i] * t
  }
  
  classified <- 0
  if (c > 0.0) {
    classified <- 1
    print(paste("c=", as.character(c),
                ". the data is classified correctly."))
  } else { # apply steepest descent method if the data is wrongly classified
    for (i in 1:n.in) {
      w[i] <- w[i] + learning.rate * x[i] * t
      print(paste("i=", as.character(i),
                  "w[1]=", as.character(w[1]),
                  "w[2]=", as.character(w[2])))
    } 
  }
  
  return(list(classified=classified, w=w)) # NOTE need to return w as well
}

# calc \sigma(wx)
PredictPerceptrons <- function(n.in, w, x) {
  # wx
  pre.activation <- 0.0
  for (i in 1:n.in) {
    pre.activation <- pre.activation + w[i] * x[i]
  }
  
  return(Step(pre.activation))
}