Salary <- read.csv("../dataset/cleanedData.csv", stringsAsFactors = TRUE)

df <- Salary
str(df)

df$Job_Type = as.factor(df$Job_Type)
df$Salary = as.factor(df$Salary)
df$python <- as.factor(df$python)
df$sql <- as.factor(df$sql)
df$machine.learning <- as.factor(df$machine.learning)
df$r <- as.factor(df$r)
df$hadoop <- as.factor(df$hadoop)
df$tableau <- as.factor(df$tableau)
df$sas <- as.factor(df$sas)
df$spark <- as.factor(df$spark)
df$java <- as.factor(df$java)
df$Others <- as.factor(df$Others)


#Splitting the dataset into training and test set with a ratio of 80 to 20
library(caTools)
set.seed(123)
split = sample.split(Salary$Salary, SplitRatio = 0.8)
training_set = subset(Salary, split == TRUE)
test_set = subset(Salary, split == FALSE)

#naive bayes
library(e1071)
library(caret)
#install.packages("caret")
bayes <- naiveBayes(Salary ~ .,data = training_set)
summary(bayes)

y_pred <- predict(bayes, newdata = test_set) 

# Confusion Matrix 
cm <- table(test_set$Salary, y_pred) 
cm 

# Model Evauation 
confusionMatrix(cm)


trainp <- predict(bayes,training_set)
train_cm <- table(trainp,training_set$Salary)
train_cm
sum(diag(train_cm))/sum(train_cm)


testp <- predict(bayes,test_set)
test_cm <- table(testp,test_set$Salary)
test_cm
sum(diag(test_cm))/sum(test_cm)



#multinomial
library(nnet)
multinomial = multinom(formula = Salary ~ .,data = training_set)
summary(multinomial)

trainpr <- predict(multinomial,training_set)
train_cma <- table(trainpr,training_set$Salary)
train_cma
sum(diag(train_cma))/sum(train_cma)


testpr <- predict(multinomial,test_set)
test_cma <- table(testpr,test_set$Salary)
test_cma
?sum(diag(test_cma))/sum(test_cma)

#svm
library(e1071)
svm <- svm(formula = Salary ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'radial')
strain <- predict(svm,training_set)
scm <- table(strain,training_set$Salary)
scm
sum(diag(scm))/sum(scm)

stest <- predict(svm,test_set)
stcm <- table(stest,test_set$Salary)
stcm
sum(diag(stcm))/sum(stcm)

#decision tree
#install.packages("rpart")
library(rpart)
dt = rpart(formula = Salary ~ .,
           data = training_set)
x <- predict(dt,training_set,type="class")
cm2 <- table(x,training_set$Salary)
sum(diag(cm2))/sum(cm2)

dtest <- predict(dt,test_set,type="class")
dtcm <- table(dtest,test_set$Salary)
dtcm
sum(diag(dtcm))/sum(dtcm)

