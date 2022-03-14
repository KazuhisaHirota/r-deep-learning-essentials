source("dataset.R")
source("multi_layer_perceptrons.R")

TestMultiLayerPerceptrons <- function() {
  
  print("set configs")
  
  set.seed(123)
  
  patterns <- 2
  
  train.N <- 4
  test.N <- 4
  
  n.in <- 2
  n.hidden <- 3
  n.out <- patterns
  
  epochs <- 100
  learning.rate <- 0.1
  
  # minibatch.size must be > 1 (meaning on-line training),
  # because a matrix sliced from minibatch.train.X cube is degenerated to a vector.
  # Our code expects minibatch.train.X[batch,,] to be a matrix.
  minibatch.size <- 2
  minibatch.N <- as.integer(train.N / minibatch.size) # number of minibatches
  
  print("make train dataset")
  train.data <- MakeXORDataset()
  train.X <- train.data$X
  train.T <- train.data$T
  
  print("make test dataset")
  test.data <- MakeXORDataset()
  test.X <- test.data$X
  test.T <- test.data$T
  
  print("make minibatches")
  minibatch.index <- sample(1:train.N) # shuffle c(1, 2, ..., train.N)
  
  minibatch.train.X <- array(0, c(minibatch.N, minibatch.size, n.in))
  minibatch.train.T <- array(0, c(minibatch.N, minibatch.size, n.out))
  for (i in 1:minibatch.N) {
    for (j in 1:minibatch.size) {
      index <- minibatch.index[(i - 1) * minibatch.size + j]
      minibatch.train.X[i, j,] <- train.X[index,] # matrix
      minibatch.train.T[i, j,] <- train.T[index,] # matrix
    }
  }
  
  classifier = MultiLayerPerceptrons$new(n.in, n.hidden, n.out)
  
  print("train")
  for (epoch in 1:epochs) {
    print(paste("epoch: ", as.character(epoch)))
    for (batch in 1:minibatch.N) {
      classifier$Train(minibatch.train.X[batch,,],
                       minibatch.train.T[batch,,],
                       minibatch.size, learning.rate)
    }
  }
  
  print("test")
  predicted.T <- matrix(0, test.N, n.out)
  for (i in 1:test.N) {
    predicted.T[i,] <- classifier$Predict(test.X[i,])
  }
  
  print("evaluate")
  confusion.matrix <- matrix(0, nrow = patterns, ncol = patterns)
  accuracy <- 0.0
  precision <- numeric(patterns)
  recall <- numeric(patterns)
  
  for (i in 1:test.N) {
    predicted <- match(1, predicted.T[i,])
    actual <- match(1, test.T[i,])
    confusion.matrix[actual, predicted] <- confusion.matrix[actual, predicted] + 1
  }
  
  for (i in 1:patterns) {
    col <- 0
    row <- 0
    
    for (j in 1:patterns) {
      if (i == j) {
        accuracy <- accuracy + confusion.matrix[i, j]
        precision[i] <- precision[i] + confusion.matrix[j, i] # NOTE
        recall[i] <- recall[i] + confusion.matrix[i, j]
      }
      
      col <- col + confusion.matrix[j, i] # NOTE
      row <- row + confusion.matrix[i, j]
    }
    
    precision[i] <- precision[i] / col
    recall[i] <- recall[i] / row
  }
  accuracy <- accuracy / test.N
  
  print("MLP model evaluation")
  print(paste("Accuracy: ", as.character(accuracy * 100.)))
  print("Precison:")
  for (i in 1:patterns) {
    print(paste("class", as.character(i), ": ", as.character(precision[i] * 100.)))
  }
  print("Recall:")
  for (i in 1:patterns) {
    print(paste("class", as.character(i), ": ", as.character(recall[i] * 100.)))
  }
}

TestMultiLayerPerceptrons()