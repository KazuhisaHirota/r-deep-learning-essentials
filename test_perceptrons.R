source("dataset.R")
source("perceptrons.R")

TestPerceptrons <- function() {
  print("set configs")
  
  set.seed(1234)
  
  train.n = 1000 # number of training data
  test.n = 200 # number of test data
  
  n.in = 2 # dim of input data
  # n.out = 1
  
  epochs = 100
  learning.rate = 1.0 # learning rate can be 1 in perceptrons
  
  print("make data set")
  
  # class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
  mu11 <- -2.0
  mu12 <- 2.0
  answer1 <- 1
  # class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
  mu21 <- 2.0
  mu22 <- -2.0
  answer2 <- -1
  # make training data
  train.data <- MakeDataset(train.n,
                            mu11, mu12, answer1,
                            mu21, mu22, answer2)
  train.x <- train.data$x
  train.t <- train.data$t
  # make test data
  test.data <- MakeDataset(test.n,
                           mu11, mu12, answer1,
                           mu22, mu22, answer2)
  test.x <- test.data$x
  test.t <- test.data$t
  
  # construct
  classifier = Perceptrons$new(n.in)
  
  # train
  print("train")
  epoch <- 0 # training epoch counter
  while(TRUE) {
    print(paste("epoch: ", as.character(epoch)))
    
    classified_ <- 0
    for (i in 1:train.n) {
      print(paste("using data train.x[", as.character(i), ",]"))
      classified_ <- classified_ +
                     classifier$Train(train.x[i,], # NOTE x[i,]
                                      train.t[i], learning.rate)
    }
    if (classified_ == train.n) { # when all data are classified correctly
      print("all data are classified correctly. break.")
      break
    }
    
    epoch <- epoch + 1
    if (epoch > epochs) {
      print("epoch > epochs. break.")
      break
    }
  }
  
  # test
  print("test")
  predicted.t = numeric(test.n) # outputs predicted by the model
  for (i in 1:test.n) {
    predicted.t[i] <- classifier$Predict(test.x[i,]) # NOTE x[i,]
  }
  
  # evaluate the model
  print("evaluate")
  confusion.matrix <- matrix(0, nrow=2, ncol=2)
  accuracy <- 0.0
  precision <- 0.0
  recall <- 0.0
  
  for (i in 1:test.n) {
    if (predicted.t[i] > 0) { # positive
      if (test.t[i] > 0) { # TP
        accuracy <- accuracy + 1
        precision <- precision + 1
        recall <- recall + 1
        # NOTE matrix[i, j], instead of matrix[i][j]
        confusion.matrix[1, 1] <- confusion.matrix[1, 1] + 1
      } else { # FP
        confusion.matrix[2, 1] <- confusion.matrix[2, 1] + 1
      }
    } else { # negative
      if (test.t[i] > 0) { # FN
        confusion.matrix[1, 2] <- confusion.matrix[1, 2] + 1
      } else { # TN
        accuracy <- accuracy + 1
        confusion.matrix[2, 2] <- confusion.matrix[2, 2] + 1
      }
    }
  }
  accuracy = accuracy / test.n # (TP + TN) / (all)
  precision = precision / (confusion.matrix[1, 1] + confusion.matrix[2, 1]) # TP / (TP + FP)
  recall = recall / (confusion.matrix[1, 1] + confusion.matrix[1, 2]) # TP / (TP + FN)
  
  print("Perceptrons model evaluation")
  print(paste("Accuracy: ", as.character(accuracy * 100)))
  print(paste("Precision: ", as.character(precision * 100)))
  print(paste("Recall: ", as.character(recall * 100)))
}

TestPerceptrons()