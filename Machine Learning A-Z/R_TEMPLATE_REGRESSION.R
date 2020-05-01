# REGRESSION TEMPLATE R

dataset = read.csv('path_file.csv')
dataset = dataset['what_we_want']

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE) 
test_set = subset(dataset,split == FALSE)

# SCALING 
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])

# FITTING THE REGRESSION MODEL TO THE DATASET
# CREATE THE REGRESSOR HERE
regressor =

# PREDICTING A NEW RESULT
y_pred = predict(regressor, data.frame(Level = 6.5))
  
# VISUALISING
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$"x",
                 y = dataset$"y"),
             colour = 'red') +
  geom_line(aes(x = dataset$Level,
                y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Title') +
  xlab('x') +
  ylab('y')


# VISUALISING SMOOTHLY
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level),0.1)
ggplot() +
  geom_point(aes(x = dataset$"x",
                 y = dataset$"y"),
             colour = 'red') +
  geom_line(aes(x = x_grid,
                y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Title') +
  xlab('x') +
  ylab('y')





