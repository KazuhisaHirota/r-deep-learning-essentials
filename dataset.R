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

MakeDataset2 <- function(data.size,
                         mu11, mu12, answer1, # answer1: vector
                         mu21, mu22, answer2, # answer2: vector
                         mu31, mu32, answer3  # answer3: vector
                         ) {
  
  X <- matrix(0, nrow=data.size, ncol=2)
  T <- matrix(0, nrow=data.size, ncol=3)
  
  one.class.size <- data.size / 3 # NOTE
  
  # class 1 data
  for (i in 1:one.class.size) {
    X[i, 1] <- rnorm(1) + mu11
    X[i, 2] <- rnorm(1) + mu12
    T[i, ] <- answer1
  }
  # class 2 data
  for (i in (one.class.size + 1):(one.class.size * 2)) {
    X[i, 1] <- rnorm(1) + mu21
    X[i, 2] <- rnorm(1) + mu22
    T[i, ] <- answer2
  }
  # class 3 data
  for (i in (one.class.size * 2 + 1):data.size) {
    X[i, 1] <- rnorm(1) + mu31
    X[i, 2] <- rnorm(1) + mu32
    T[i, ] <- answer3
  }
  
  return(list(X=X, T=T))
}

MakeXORDataset <- function() {
  #X = np.array([
  #  [0., 0.],
  #  [0., 1.],
  #  [1., 0.],
  #  [1., 1.]
  #])
  X <- rbind(
    c(0., 0.),
    c(0., 1.),
    c(1., 0.),
    c(1., 1.)
  )
  
  #T = np.array([
  #  [0, 1],
  #  [1, 0],
  #  [1, 0],
  #  [0, 1]
  #])
  T <- rbind(
    c(0., 1.),
    c(1., 0.),
    c(1., 0.),
    c(0., 1.)
  )
  
  return(list(X=X, T=T))
}







