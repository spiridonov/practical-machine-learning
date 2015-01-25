library(caret)
library(doMC)
registerDoMC(cores = 5)

training = read.csv("pml-training.csv")
testing = read.csv("pml-testing.csv")

pml_write_files <- function(x) {
  n <- length(x)
  for(i in 1:n){
    filename <- paste0("problem_id_",i,".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}

lotOfNAs <- function(vector) {
  if(sum(is.na(vector)) / length(vector) > 0.9) {
    res <- TRUE                        
  } else {                                      
    res <- FALSE
  }
  invisible(res)
}

sanitizeData <- function(data) {
  result <- data[ , -(1:7)]
  nums <- sapply(result, is.numeric)
  result <- result[ , nums]  
  
  varsWith90NAs <- sapply(result, lotOfNAs)
  result <- result[ , !varsWith90NAs]
  
  invisible(result)
}

sanitizedTraining <- sanitizeData(training)
sanitizedTraining$classe <- training$classe

subsetTrainingIndex <- createDataPartition(sanitizedTraining$classe, p = 0.90, list = FALSE)
subsetTraining <- sanitizedTraining[subsetTrainingIndex, ]
subsetTesting <- sanitizedTraining[-subsetTrainingIndex, ]

last <- ncol(subsetTraining)
pca <- preProcess(subsetTraining[ , -last], method = "pca", pcaComp = 5)
trainPC <- predict(pca, subsetTraining[ , -last])
tc = trainControl(allowParallel = TRUE, method = "cv", number = 5)
modelFit <- train(subsetTraining$classe ~ ., method = "rf", trainControl = tc, data = trainPC)

last <- ncol(subsetTesting)
testPC <- predict(pca, subsetTesting[ , -last])
prediction <- predict(modelFit, testPC)
confusionMatrix(prediction, subsetTesting$classe)

last <- ncol(testing)
valid <- sanitizeData(testing[ , -last])
validPC <- predict(pca, valid)
pred_valid <- predict(modelFit, validPC)

pml_write_files(pred_valid)

