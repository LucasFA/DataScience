rm(list = ls())
lista.scores <- c()
# Feature data
features_sucio <- read.csv("./H1N1/training_set_features.csv", as.is = F)
# str(features)
noFactores <- which(colnames(features_sucio) %in% c("respondent_id", "household_adults", "household_children"))
columnasFactores <- colnames(features_sucio)[-noFactores]
# str(features[columnasFactores])

features_sucio[columnasFactores] <- lapply(features_sucio[columnasFactores], as.factor)
# str(features[columnasFactores])

rm(list = c("columnasFactores", "noFactores"))
# --------------------------------------------------------------------------------
# Label data
labels_sucio <- read.csv("./H1N1/training_set_labels.csv")


# --------------------------------------------------------------------------------
# Cleaning data
completeCases <- complete.cases(features_sucio)
# Throwing away half the data...
# TODO: get a better solution
features_limpiado <- features_sucio[completeCases, ]
labels_limpiado <- labels_sucio[completeCases, ]
data.nrows <- length(features_sucio[, 1])
rm(completeCases)
# TODO: habrá que predecir valores cuando no esté todo lleno también
# --------------------------------------------------------------------------------
# Training and test splits
ratio <- 0.75
set.seed(1759)
indices <- sample.int(n = data.nrows)
indices.train <- sample(indices, size = round(data.nrows * ratio))
indices.test <- sample(indices, size = round(data.nrows * (1 - ratio)))
rm(list = c("ratio", "data.nrows"))

# NOTE: Using dirty data
train <- list(
    data = subset(features_sucio, select = -c(respondent_id))[indices.train, ],
    labels = subset(labels_sucio, select = -c(respondent_id))[indices.train, ]
)

test <- list(
    data = subset(features_sucio, select = -c(respondent_id))[indices.test, ],
    labels = subset(labels_sucio, select = -c(respondent_id))[indices.test, ]
)

rm(list = c(
    "indices", "indices.train", "indices.test",
    "features_limpiado", "features_sucio", "labels_limpiado", "labels_sucio"
))
# --------------------------------------------------------------------------------
# Modeling

# --------------------------------------------------------------------------------
# XGBOOST

library(xgboost)
library(AUC)

createDMatrix <- function(data, labels) {
    xgb.DMatrix(
        data = data.matrix(data),
        label = labels
    )
}

dtrain <- list(
    h1n1 = createDMatrix(train$data, train$labels$h1n1_vaccine),
    seasonal = createDMatrix(train$data, train$labels$seasonal)
)

dtest <- list(
    h1n1 = createDMatrix(test$data, test$labels$h1n1_vaccine),
    seasonal = createDMatrix(test$data, test$labels$seasonal)
)

watchlist <- list()
watchlist$h1n1 <- list(train = dtrain$h1n1, test = dtest$h1n1)
watchlist$seasonal <- list(train = dtrain$seasonal, test = dtest$seasonal)
rm(list = c("train"))

bstSparse <-
    list(
        h1n1 = xgb.train(
            data = dtrain$h1n1,
            max.depth = 20L,
            eta = 0.5,
            subsample = 0.50,
            colsample_bytree = 0.5,
            num_parallel_tree = 16L,
            nrounds = 101L,
            watchlist = watchlist$h1n1,
            objective = "binary:logistic",
            nthread = 8L,
            print_every_n = 5L
        ),
        seasonal = xgb.train(
            data = dtrain$seasonal,
            max.depth = 24L,
            eta = 0.5,
            subsample = 0.50,
            colsample_bytree = 0.5,
            num_parallel_tree = 16L,
            nrounds = 121L,
            watchlist = watchlist$seasonal,
            objective = "binary:logistic",
            nthread = 8L,
            print_every_n = 5L
        )
    )

# Check XGBOOST test output
pred <- list(
    h1n1 = predict(bstSparse$h1n1, dtest$h1n1),
    seasonal = predict(bstSparse$seasonal, dtest$seasonal)
)

prediction <- lapply(pred, function(x) as.numeric(x > 0.5))
err <- list(
    h1n1 = mean(prediction$h1n1 != test$labels$h1n1_vaccine),
    seasonal = mean(prediction$seasonal != test$labels$seasonal)
)

paste("test-error=", err)
scores <- c(
    auc(roc(pred$h1n1, factor(test$labels$h1n1_vaccine))),
    auc(roc(pred$seasonal, factor(test$labels$seasonal)))
)
lista.scores <- c(lista.scores, mean(scores))
tail(lista.scores)

# Continuar entrenamiento con modelo anterior
# Hecho una vez
# bstSparse$h1n1 <-
#     xgboost(
#         data = dtrain$h1n1,
#         xgb_model = bstSparse$h1n1,
#         max.depth = 22L,
#         eta = 0.1,
#         subsample = 0.5,
#         nthread = 8L,
#         nrounds = 41L,
#         objective = "binary:logistic",
#         print_every_n = 10L
#     )

# bstSparse$seasonal <-
#     xgboost(
#         data = dtrain$seasonal,
#         xgb_model = bstSparse$seasonal,
#         max.depth = 22L,
#         eta = 0.1,
#         subsample = 0.5,
#         nthread = 8L,
#         nrounds = 81L,
#         objective = "binary:logistic",
#         print_every_n = 10L
#     )


# Es un bueno modelo? Pues ahora  probamos con todo el dataset

# Idea 1. Imputar tomando la media del valor
# Idea 1.1. Imputar con algún otro valor que haga que la variable no afecte a

# Idea 2.


features_test <- read.csv("./H1N1/test_set_features.csv")

noFactores <- which(colnames(features_test) %in% c("respondent_id", "household_adults", "household_children"))
columnasFactores <- colnames(features_test)[-noFactores]

features_test[columnasFactores] <- lapply(features_test[columnasFactores], as.factor)

rm(list = c("columnasFactores", "noFactores"))



d_to_predict <- features_test |>
    subset(select = -c(respondent_id)) |>
    data.matrix()

d_to_predict <- list(
    h1n1 = xgb.DMatrix(d_to_predict),
    seasonal = xgb.DMatrix(d_to_predict)
)

pred <- list(
    h1n1 = predict(bstSparse$h1n1, d_to_predict$h1n1),
    seasonal = predict(bstSparse$seasonal, d_to_predict$seasonal)
)


prediction <- lapply(pred, function(x) as.numeric(x > 0.5))

csv_a_escribir <- cbind(
    features_test$respondent_id,
    pred$h1n1,
    pred$seasonal
)


header <- c("respondent_id", "h1n1_vaccine", "seasonal_vaccine")

write.table(
    csv_a_escribir,
    file = "resultados.csv",
    col.names = header,
    row.names = FALSE,
    quote = FALSE,
    sep = ","
)
