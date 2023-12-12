library(tidymodels)
library(embed)
library(vroom)
library(tidyverse)
library(doParallel)
library(skimr)
library(DataExplorer)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

acs_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/Allstate-Claims-Severity/train.csv',
                      show_col_types = FALSE)
acs_test <- vroom('C:/BYU/2023(5) Fall/STAT 348/Allstate-Claims-Severity/test.csv',
                   show_col_types = FALSE)

acs_cont <- acs_train %>%
  select(118:132)

skimr::skim(acs_train)
DataExplorer::plot_correlation(acs_cont)
  
