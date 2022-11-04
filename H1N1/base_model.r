rm(list = ls())
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
data.nrows <- features.nrows <- labels.nrows <- length(features_limpiado[, 1])
# TODO: habrá que predecir valores cuando no esté todo lleno también
# --------------------------------------------------------------------------------
# Training and test splits
ratio <- 0.9
set.seed(127)
indices <- sample.int(n = data.nrows)
indices.train <- sample(indices, size = round(data.nrows * ratio))
indices.test <- sample(indices, size = round(data.nrows * (1 - ratio)))

train <- list(
    data = subset(features_limpiado, select = -c(respondent_id))[indices.train, ],
    labels = subset(labels_limpiado, select = -c(respondent_id))[indices.train, ]
)

test <- list(
    data =  subset(features_limpiado, select = -c(respondent_id))[indices.test, ],
    labels = subset(labels_limpiado, select = -c(respondent_id))[indices.test, ]
)

# ────────────────────────────────────────────────────────────────────────────────
# Modeling

## BASE MODEL
all.features.string <- function(explained) {
    step1 <- paste(names(features_limpiado[2:length(features_limpiado)]), collapse = "+")[[1]]
    step2 <- c("~", step1)
    step3 <- paste(step2, collapse = "")

    paste0(c(explained, step3), collapse = "")
}

# Model h1n1
formula_h1n1.string <- all.features.string("h1n1_vaccine")
formula_h1n1 <- formula(formula_h1n1.string)
model_h1n1 <- lm(formula = formula_h1n1, data = cbind(train$data, train$labels))

summary(model_h1n1)

pred <- predict(model_h1n1, test$data)
prediction <- as.numeric(pred > 0.5)
err <- mean(prediction != test$labels)
print(paste("test-error=", err))


model_h1n1.reduced <- step(model_h1n1)
summary(model_h1n1.reduced)

pred <- predict(model_h1n1.reduced, test$data)
prediction <- as.numeric(pred > 0.5)
err <- mean(prediction != test$labels)
print(paste("test-error=", err))
# test error: 0.2739
