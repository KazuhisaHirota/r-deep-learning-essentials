MakeDataset <- function(data.size,
                        mu11, mu12, answer1,
                        mu21, mu22, answer2) {
  
  x = matrix(0, nrow=data.size, ncol=2) # input data for training
  t = numeric(data.size) # answers (labels) for training
  
  one.class.size = data.size / 2
  
  # class 1 data
  for (i in 1:one.class.size) {
    x[i, 1] <- rnorm(1) + mu11 # input variable 1
    x[i, 2] <- rnorm(1) + mu12 # input variable 2
    t[i] <- answer1
  }
  
  # class 2 data
  for (i in (one.class.size+1):data.size) { # NOTE "()" is necessary
    x[i, 1] <- rnorm(1) + mu21 # input variable 1
    x[i, 2] <- rnorm(1) + mu22 # input variable 2
    t[i] <- answer2
  }
  
  return(list(x=x, t=t)) # NOTE return results with names
}