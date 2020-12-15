# load necessary packages -----------------------------------------------------
library(tidymodels)
library(stacks)
library(tibble)

# "load" in basil data ;) -----------------------------------------------------
basil <- 
  tibble(
    height = rnorm(500, 10, 1),
    species = 
      sample(
        c("sweet", "genovese", "napoletano"), 
        500, 
        TRUE
      ),
    chemical_lol = rnorm(500, 1, .5),
    mass = height + chemical_lol + rnorm(500, 0, 2)
  )

# define model specifications -------------------------------------------------
folds <- vfold_cv(basil, v = 5)

basil_rec <-
  recipe(mass ~ ., basil) %>%
  step_dummy(all_nominal()) %>%
  step_zv(all_predictors())

lr_res <-
  fit_resamples(
    workflow() %>%
      add_recipe(basil_rec) %>%
      add_model(linear_reg() %>% set_engine("lm")),
    resamples = folds,
    control = control_stack_resamples(),
    metrics = metric_set(rmse)
  )

knn_res <- 
  tune_grid(
    workflow() %>%
      add_recipe(basil_rec) %>%
      add_model(
        nearest_neighbor(
          neighbors = tune()) %>% 
          set_engine("kknn") %>% 
          set_mode("regression")
      ),
    resamples = folds,
    grid = 4,
    control = control_stack_grid(),
    metrics = metric_set(rmse)
  )

nn_res <- 
  tune_grid(
    workflow() %>%
      add_recipe(basil_rec) %>%
      add_model(
        mlp(
          hidden_units = tune(), 
          dropout = tune()) %>% 
          set_engine("keras") %>% 
          set_mode("regression")
      ),
    resamples = folds,
    grid = 6,
    control = control_stack_grid(),
    metrics = metric_set(rmse)
  )

# check out the model specifications ------------------------------------------
lr_res

knn_res

nn_res

# build the ensemble ----------------------------------------------------------
st <- stacks() %>%
  add_candidates(lr_res) %>%
  add_candidates(knn_res) %>%
  add_candidates(nn_res) %>%
  blend_predictions() %>%
  fit_members()

st