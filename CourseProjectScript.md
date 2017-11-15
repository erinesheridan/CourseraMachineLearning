## Coursera Machine Learning - Course Project

#set the working directory to where the data files are located

setwd("C:/Users/Esherida/Documents/R/Coursera/MachineLearning")

#call on the libraries that will be needed

library(caret)

library(ggplot2)
library(randomForest)

#Next, read the training and testing data from the files already downloaded from
#the website

training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

#Take a look at the data provided (training only!) so we know more about how to build our model.
#The goal of the model is to use any variables provided to predict the manner
#in which a person did the exercise (classe variable tells us this- classes A through E).

names(training)
str(training)
summary(training)
summary(training$classe) # this is the variable/outcome we want to predict from the model

#Before doing anything else, we will set aside a subset of our training data for
#cross-validation (40% for cross-validation, 60% to train on).
#Remember, we are going to predict the variable "classe" using the other variables to predict

inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTrain <- training[inTrain, ]
myTest <- training[-inTrain, ]

#to see the dimensions of our training subset (which we will train on)

dim(myTrain)

#to see the dimensions of our testing subset (which we will cross-validate on)

dim(myTest)

#to create a scatter plot matrix to view the data:

featurePlot(x=training[, 150:159], y = training$classe, plot = 'pairs')

#we know from looking at the data that there are some variables that will not be useful
#to us. There are some variables that are mainly missing (NA), as well as some variables
#that do not have any relation to the outcome variable, for example, the 1st column (x),
#which is just an index of the row number. There also may be variables with very little
#variance, which would also not be useful to us.

#first, let's remove the variables that are mostly missing
#we will make our threshold greater than 60%:

mytrain_subset <- myTrain

#make a new dataframe that we will edit and remove columns from,
#so the original still exists

for (i in 1:length(myTrain)) {
  if (sum(is.na(myTrain[ , i])) / nrow(myTrain) > .60) {
    for (j in 1:length(mytrain_subset)) {
      if (length(grep(names(myTrain[i]), names(mytrain_subset)[j]))==1) {
        mytrain_subset <- mytrain_subset[ , -j]
      }
    }
  }
}

#to see the dimensions of our subset - and how many columns/variales we removed
#we removed 100 variables (160 - 60 = 100)
dim(mytrain_subset)

#to see which variables are left:
names(mytrain_subset)

#next, let's get rid of the variables that are obviously not predictors 
#like the time of day they were performing the exercise, and their name. While time of day
#might help predict the correctness/incorrectness of an exercise in a different study (
#for example, if more fatigued people perform the exercise worse), this study had persons
#purposely performing the exercised incorrectly so the machines could measure their body -
#the the persons name, the time of day, and the window is irrelevant here.
#these are the first 7 variables, so we will keep the rest:
mytrain_subset2 <- mytrain_subset[,8:length(mytrain_subset)]

#Next, we will remove all variables with near zero variance
NearZeroV <- nearZeroVar(mytrain_subset2, saveMetrics = TRUE)
NearZeroV # they are all false, so there are none to remove

#Looks like 'mytrain_subset2' is the final dataset to do our model training on:
dim(mytrain_subset2)
names(mytrain_subset2)
#we have 52 variables to use to predict the one variable, 'classe'

#Time to build the model
#We will use Random Forest as our machine learning alghorithem because, as one of the class
#lectures mentioned, "it's one of the most widely used and highly accurate methods
#for prediction"
#also because I am have a problem installing "rattle" (doesn't exist for R version 3.4.2?)
#and therefore cannot do a fancyRpartPlot
#and some of the other commands from the rattle library

#from lecture:

modFit <- train(classe~., data=mytrain_subset2, method="rf", prox=TRUE)

#this makes a bunch of different trees
#can look at specific trees in the model with:
#getTree(modFit$finalModel, k=2)
#this gives me the second tree (k=2)

#Let's try the 'randomForest' function instead

set.seed(291989)

#so the results can be reproduced by me and others

modFit <- randomForest(classe ~ ., data = mytrain_subset2)
modFit

#cross validation on the testing data (40% subset of original training set, put aside
#for cross-validation)
#the accuracy here will also tell me my expected out of sample error

predict1 <- predict(modFit, myTest, type = "class")
confusionMatrix(myTest$classe, predict1)

#accuracy is 99.35%, so that can be expected to be inform my out-of-sample error
#(100-99.35 = 0.65%), considering
#my cross-validation set is essentially "out of sample" because it wasn't used for any
#of the training

#just out of curiosity, what was my in sample erro?
#in sample error would be gleaned from the accuracy of the training set, which was used to
#train the model - we would expect this to be slightly over-fit, and for the out of sample
#error to be greater because the model was training on this exact data

predict_train <- predict(modFit, myTrain, type = "class")
confusionMatrix(myTrain$classe, predict_train)

#another way to do the same thing - not using "randomForest":

modFit2 <- train(classe ~., method = "rf", trControl=trainControl(method = "cv", number = 4), data = mytrain_subset2)

#The final step is to apply the model to the test set, which was never used at all for
#anything. These 20 cases were set aside at the outset, read from their own csv file.

predict_FINAL <- predict(modFit, testing, type = "class")
predict_FINAL

#this prints the prediction for the 20 unknown test cases, which should be around 99% accurate
#(after submitting the 20 cases to the quiz, all 20 were correct)

require(knitr)
require(markdown)
knit("CourseProjectScript.Rmd")
markdownToHTML('CourseProjectScript.md', 'CourseProjectScript.html', options=c("use_xhml"))
