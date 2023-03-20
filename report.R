---
title: "Summative assignment for ASML Classification"
author: "Kolotioloma Soro"
---

```{r}
require(dplyr)
bankloans = read.csv("bank_personal_loan.csv", header=TRUE)
head(bankloans)
bankloans$Personal.Loan = factor(bankloans$Personal.Loan)
```

```{r}
skimr::skim(bankloans)
colnames(bankloans)
plot(bankloans)
length(colnames(bankloans))
```

---
EXPLORATORY ANALYSIS
---


```{r}
library("GGally")
library(ggplot2)

# Visualising the relationship between some non-binary features
ggpairs(bankloans |> dplyr::select(Personal.Loan, Age ,Experience, Income,Family ), aes(color = factor(Personal.Loan)))
```

```{r}
grouped_df <- bankloans %>% 
  group_by(Personal.Loan) %>% 
  count(Personal.Loan)
grouped_df
```

```{r}
grouped_Age <- bankloans %>%
  group_by(bankloans$Age) %>%
  summarize(mean_value = mean(Income))
grouped_Age

ggplot(grouped_Age |> arrange(desc(`bankloans$Age`)),
       aes(x = `bankloans$Age`, y = mean_value)) +
  geom_col()

bankloans
```

```{r}
bankloans

DataExplorer::plot_bar(bankloans, ncol = 5)
DataExplorer::plot_histogram(bankloans, ncol = 5)
DataExplorer::plot_boxplot(bankloans, by = "Personal.Loan", ncol = 5)

```

```{r}

library(zipcodeR)

zips <- rep(0,5000)
# zips <- list()
for (i in 1:5000){
  
  tryCatch({
  zips[[i]] = reverse_zipcode(bankloans$ZIP.Code[[i]])$zipcode_type
}, error = function(e) {
  zips[[i]] = "Others"

})

}
f_zips = zips 
f_zips = as.factor(f_zips)

```


```{r}
# View will open a tab where you can more easily see the detail ...
View(as.data.table(mlr_learners))
# ... whilst this just gives the names
mlr_learners

# Again, to see help on any of them, prefix the key name with mlr_learners_
?mlr_learners_classif.log_reg
```


```{r}
library("data.table")
library("mlr3verse")

set.seed(212) # set seed for reproducibility
bankloans_classif <- TaskClassif$new(id = "bankloans",
                               backend = bankloans, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                               positive = "0")


# We will 5-fold cross-validation as our resampling technique
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bankloans_classif)

```

We will choose a baseline classifier to be the classif.featureless classifier, and make sure to beat this performance in our final model choice.
Added to that we will use the following classifiers:
  *classif.rpart, with a penality term that we will determine using cross validation
  *classif.xgboost
  *classif.log_reg
  *classif.kknn
  *classif.ranger
  *classif.lda
  *classif.log_reg

```{r}
# Determining the penality term for our classif.rpart model
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
# lrn_enco <- po("encode") %>>%
#   po(lrn_cart_cv)

res_cart_cv <- resample(bankloans_classif, lrn_cart_cv, cv5, store_models = TRUE)

rpart::plotcp(res_cart_cv$learners[[4]]$model)

# We will use the mean of the 5 tree learners as out estimate of cp
mean(c(0.048,0.057,0.053,0.034))
```

we will use 0.048 as cp

```{r}
# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda      <- lrn("classif.lda", predict_type = "prob")
lrn_kknn    <- lrn("classif.kknn", predict_type = "prob") 
```


```{r}
# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

```

Fitting all the models above and then comparing the performances

```{r}

lrn_cart_cp_0  <- lrn("classif.rpart", predict_type = "prob")

pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)

pl_lrn_ranger <- pl_missing %>>%
  po(lrn_ranger)

pl_lrn_lda <- pl_missing %>>%
  po(lrn_lda)

pl_lrn_kknn <- pl_missing %>>%
  po(lrn_kknn)

pl_lrn_xgboost <- po("encode") %>>%
  po(lrn_xgboost)


res <- benchmark(data.table(
  task       = list(bankloans_classif),
  learner    = list(lrn_baseline,
                    lrn_cart_cp_0,
                    pl_log_reg,
                    pl_lrn_ranger,
                    pl_lrn_lda,
                    pl_lrn_kknn,
                    pl_lrn_xgboost),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


```


Adding the new zipcode column and repeating the analysis
```{r}

banklaon.zip <-bankloans
banklaon.zip$ZIP.Code <- f_zips



bankloans_classif2 <- TaskClassif$new(id = "bankloans",
                               backend = banklaon.zip, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                               positive = "0")
cv6 <- rsmp("cv", folds = 5)
cv6$instantiate(bankloans_classif2)

res2 <- benchmark(data.table(
  task       = list(bankloans_classif2),
  learner    = list(lrn_baseline,
                    lrn_cart_cp,
                    pl_log_reg,
                    pl_lrn_ranger,
                    pl_lrn_lda,
                    pl_lrn_kknn,
                    pl_lrn_xgboost),
  resampling = list(cv6)
), store_models = TRUE)

res2$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
```


Removing the zipcode col, since no major improvement and unsing the 0.035 cp value for our rpart model.
Then creating a superlearner

```{r}
# Define a super learner

bankloans <- bankloans %>% dplyr::select(-ZIP.Code)
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob",cp = 0.035, id = "cartcp")

lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")



set.seed(212) # set seed for reproducibility
bankloans_classif.last <- TaskClassif$new(id = "bankloans",
                               backend = bankloans, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                               positive = "0")


# We will 5-fold cross-validation as our resampling technique
cv.final <- rsmp("cv", folds = 5)
cv.final$instantiate(bankloans_classif.last)


# Now define the full pipeline
spr_lrn <- gunion(list(
  
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart_cp)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("learner_cv", lrn_lda),
      po("learner_cv", lrn_kknn),

      po("nop") # This passes through the original features adjusted for
                # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

# This plot shows a graph of the learning pipeline
spr_lrn$plot()

# Finally fit the base learners and super learner and evaluate
res_spr <- resample(bankloans_classif.last, spr_lrn, cv.final, store_models = TRUE)
```

```{r}
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr"),
                       msr("classif.auc")
                      
                       ))
```


fitting the chosen rpart model and visualing the performance

```{r}
library(mlr3viz)
library(mlr3benchmark)

set.seed(23)
autoplot(res_spr, measure = msr("classif.acc"), type = "boxplot")

splits = partition(bankloans_classif, ratio = 0.8)


lrn_cart_cp$train(bankloans_classif, splits$train)

pred = lrn_cart_cp$predict(bankloans_classif, splits$test)
pred$confusion


mlr3measures::confusion_matrix(truth = pred$truth,
  response = pred$response, positive = bankloans_classif.last$positive)

```


```{r}

thresholds = sort(pred$prob[,1])

rocvals = data.table::rbindlist(lapply(thresholds, function(t) {
  pred$set_threshold(t)
  data.frame(
    threshold = t,
    FPR = pred$score(msr("classif.fpr")),
    TPR = pred$score(msr("classif.tpr"))
  )
}))

head(rocvals)

```

```{r}
library(ggplot2)
autoplot(pred, type = "roc")
```

```{r}
autoplot(pred, type = "threshold", measure = msr("classif.fpr"))
autoplot(pred, type = "threshold", measure = msr("classif.acc"))
```

```{r}
autoplot(pred, type = "prc")
```

