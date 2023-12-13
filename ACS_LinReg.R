library(tidymodels)
library(embed)
library(vroom)
library(tidyverse)


acs_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/Allstate-Claims-Severity/train.csv',
                   show_col_types = FALSE) %>%
  mutate_at(vars(cat1:cat116), as.factor)

acs_test <- vroom('C:/BYU/2023(5) Fall/STAT 348/Allstate-Claims-Severity/test.csv',
                  show_col_types = FALSE) %>%
  mutate_at(vars(cat1:cat116), as.factor)

acs_train$loss <- (acs_train$loss +1)^.25

my_recipe <- recipe(loss ~ ., acs_train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) 

preg_model <- linear_reg() %>%
  set_engine("lm")

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(acs_train) 

preg_Wf%>%
  predict(new_data = acs_test)

Sub1 <- preg_wf %>%
  bind_cols(acs_test) %>%
  select(id,.pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = exp(loss))

vroom_write(Sub1, file= './submission.csv', delim=',')
