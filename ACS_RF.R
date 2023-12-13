##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(themis)
library(vroom)

# Read in Data
train <- vroom('/kaggle/input/allstate-claims-severity/train.csv')
test <- vroom('/kaggle/input/allstate-claims-severity/test.csv')

#transform into log
train$loss <- log(train$loss) 

#Creat Recipe
rf_recipe <- recipe(loss ~ ., data=train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))


#Set up the model
my_mod <- rand_forest(mtry = 5,
                      min_n=2,
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Create a workflow with model & recipe
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(my_mod) %>%
  fit(data=train)

#Predict with workflow
rf_predictions <- rf_wf %>%
  predict(new_data = test)

#Format predictions to fit Kaggle requirments
Sub1 <- rf_predictions %>%
  bind_cols(test) %>%
  select(id,.pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = exp(loss))

#Write csv file
vroom_write(Sub1, file= './submission.csv', delim=',')