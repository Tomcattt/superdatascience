# Data Preprocessing

# Importing the dataset
dataset = read.csv('data.csv')
dataset = dataset[, 2:3]

# Spliting
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE) 
test_set = subset(dataset,split == FALSE) 

# Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])