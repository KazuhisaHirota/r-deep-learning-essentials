source("dataset.R")
source("logistic_regression.R")

TestLogisticRegression <- function() {
  
  print("set configs")
  
  set.seed(1234)
  
  patterns <- 3 # number of classes
  
  train.size <- 400
  train.N <- train.size * patterns
  
  test.size <- 60
  test.N <- test.size * patterns
  
  n.in <- 2
  n.out <- patterns
  
  epochs <- 100
  learning.rate <- 0.2
  
  minibatch.size <- 50 # number of data in each minibatch
  minibatch.N <- as.integer(train.N / minibatch.size) # number of minibatches
  
  # class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
  mu11 <- -2.0
  mu12 <- 2.0
  answer1 <- c(1, 0, 0)
  # class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
  mu21 <- 2.0
  mu22 <- -2.0
  answer2 <- c(0, 1, 0)
  # class3 inputs x31 and x32: x31 ~ N(0.0, 1.0), x32 ~ N(0.0, 1.0)
  mu31 <- 0.0
  mu32 <- 0.0
  answer3 <- c(0, 0, 1)
  
  print("make train dataset")
  train.data <- MakeDataset2(train.N,
                             mu11, mu12, answer1,
                             mu21, mu22, answer2,
                             mu31, mu32, answer3)
  train.X <- train.data$X
  train.T <- train.data$T
  
  print("make test dataset")
  test.data <- MakeDataset2(test.N,
                            mu11, mu12, answer1,
                            mu21, mu22, answer2,
                            mu31, mu32, answer3)
  test.X <- test.data$X
  test.T <- test.data$T
  
  print("make minibatches")
  
  minibatch.train.X <- array(0, c(minibatch.N, minibatch.size, n.in))
  minibatch.train.T <- array(0, c(minibatch.N, minibatch.size, n.out))
  
  minibatch.index <- sample(1:train.N) # shuffle c(1, 2, ..., train.N)
  
  for (i in 1:minibatch.N) {
    for (j in 1:minibatch.size) {
      index <- minibatch.index[(i - 1) * minibatch.size + j]
      minibatch.train.X[i, j,] <- train.X[index,] # matrix
      minibatch.train.T[i, j,] <- train.T[index,] # matrix
    }
  }
  
  print("train")
  W <- matrix(0, nrow=n.out, ncol=n.in)
  b <- numeric(n.out)
  for (epoch in 1:epochs) {
    print(paste("epoch: ", as.character(epoch)))
    for (batch in 1:minibatch.N) {
      result <- TrainLogisticRegression(n.in, n.out, W, b,
                                        minibatch.train.X[batch,,], # cube
                                        minibatch.train.T[batch,,], # cube
                                        minibatch.size, learning.rate)
      W <- result$W
      b <- result$b
    }
    learning.rate <- learning.rate * 0.95
  }
  
  print("test")
  predicted.T <- matrix(0, test.N, n.out)
  for (i in 1:test.N) {
    predicted.T[i,] <- PredictLogisticRegression(n.in, n.out, W, b, test.X[i,])
  }
  
  print("evaluate")
  confusion.matrix <- matrix(0, nrow=patterns, ncol=patterns)
  accuracy <- 0.0
  precision <- numeric(patterns)
  recall <- numeric(patterns)
  
  for (i in 1:test.N) {
    predicted <- match(1, predicted.T[i,]) # matrix
    actual <- match(1, test.T[i,]) # matrix
    confusion.matrix[actual, predicted] <- confusion.matrix[actual, predicted] + 1
  }
  
  for (i in 1:patterns) {
    col <- 0.0
    row <- 0.0
    
    for (j in 1:patterns) {
      if (i == j) {
        accuracy <- accuracy + confusion.matrix[i, j]
        precision[i] <- precision[i] + confusion.matrix[j, i] # NOTE
        recall[i] <- recall[i] + confusion.matrix[i, j]
      }
      
      col <- col + confusion.matrix[j, i] # NOTE
      row <- row + confusion.matrix[i, j]
    }
    
    precision[i] = precision[i] / col # TP / (TP + FP)
    recall[i] = recall[i] / row # TP / (TP + FN)
  }
  accuracy = accuracy / test.N # (TP + TN) / (all)
  
  print("Perceptrons model evaluation")
  print(paste("Accuracy: ", as.character(accuracy * 100)))
  print("Precision:")
  for (i in 1:patterns) {
    print(as.character(precision[i] * 100))
  }
  print("Recall:")
  for (i in 1:patterns) {
    print(as.character(recall[i] * 100))
  }
  
}

TestLogisticRegression()