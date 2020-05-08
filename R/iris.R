# the libraries
library(caret)
library(datasets)

# Import iris
data(iris)
summary(iris)
dataset <- iris

# Split into validation and training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
validation <- dataset[-validation_index,]
dataset <- dataset[validation_index,]

# survey the dataset
dim(dataset)
sapply(dataset, class)
head(dataset)
levels(dataset$Species)

# Summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)
summary(dataset)

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# plots
par(mfrow=c(1,4))
  for (i in 1:4) {
    boxplot(x[,i], main=names(iris)[i])
  }

# bar plot
plot(y)

# multivariate plots
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plot
featurePlot(x=x, y=y, plot="box")

# density plots
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# 10-fold cross-validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

# LDA
set.seed(7)
fit.lda <- train(Species~., data = dataset, method = "lda", metric = metric, trControl = control)
# CART
set.seed(7)
fit.cart <- train(Species~., data = dataset, method = "rpart", metric = metric, trControl = control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data = dataset, method = "knn", metric = metric, trControl = control)
# SVM
set.seed(7)
fit.svm <- train(Species~., data = dataset, method = "svmRadial", metric = metric, trControl = control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data = dataset, method = "rf", metric = metric, trControl = control)


# comparison
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare models
dotplot(results)

# summarize
print(fit.lda)

# predictions
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

