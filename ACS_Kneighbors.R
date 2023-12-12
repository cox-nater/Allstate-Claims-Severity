library(tidymodels)
library(embed)
library(vroom)
library(tidyverse)
library(kknn)
library(doParallel)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

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

#prepped_recipe <- prep(my_recipe)
#baked <- bake(prepped_recipe, new_data = acs_train)

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_engine('kknn') %>%
  set_mode("regression")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)


tuning_grid <- grid_regular(neighbors(),
                            levels = 3)

folds <- vfold_cv(acs_train, v = 5, repeats = 1)

CV_results <-knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae)) 

bestTune <- CV_results %>%
  select_best("mae")

final_knn_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = acs_train)

final_wf %>% 
  predict(new_data = acs_train)

Sub1 <- final_wf %>%
  bind_cols(acs_test) %>%
  select(id,.pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = (loss^4)-1)
