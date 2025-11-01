# main.R — Titanic Survival Prediction (R version)

library(tidyverse)
library(caret)
options(warn = -1)  # suppress warnings like Python's warnings.filterwarnings()

cat("Loading Titanic dataset...\n")

train <- read.csv("src/data/train.csv")
test  <- read.csv("src/data/test.csv")
cat("Data loaded successfully!\n")

# === Explore and Adjust Data ===
cat("\nExploring and adjusting data...\n")

# Encode 'Sex' column
train$Sex <- ifelse(train$Sex == "male", 0, 1)
test$Sex  <- ifelse(test$Sex == "male", 0, 1)
cat("Converted 'Sex' column to numeric.\n")

# Impute missing values for Age and Fare
for (col in c("Age", "Fare")) {
  train[[col]][is.na(train[[col]])] <- mean(train[[col]], na.rm = TRUE)
  test[[col]][is.na(test[[col]])]   <- mean(test[[col]], na.rm = TRUE)
}
cat("Filled missing values in 'Age' and 'Fare' with mean values.\n")

# === Feature Selection ===
X_cols <- c("Pclass", "Age", "Sex", "Fare")

X <- train[, X_cols]
y <- train$Survived

# === Split Data (80/20) ===
set.seed(42)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
X_test  <- X[-train_indices, ]
y_train <- y[train_indices]
y_test  <- y[-train_indices]
cat("Data split into training and testing sets.\n")

# === Train Logistic Regression Model ===
cat("\nTraining logistic regression model...\n")
model <- glm(y_train ~ ., data = X_train, family = binomial())
cat("Model training complete.\n")

# === Evaluate Model ===
train_pred <- ifelse(predict(model, X_train, type = "response") > 0.5, 1, 0)
test_pred  <- ifelse(predict(model, X_test, type = "response") > 0.5, 1, 0)

train_acc <- mean(train_pred == y_train)
test_acc  <- mean(test_pred == y_test)

cat(sprintf("Training accuracy: %.3f\n", train_acc))
cat(sprintf("Test accuracy: %.3f\n", test_acc))
cat("Model training complete.\n")

# === Predict on Full Test Set ===
cat("\nPredicting on test set...\n")
X_test_final <- test[, X_cols]

# Predict survival
final_pred <- ifelse(predict(model, X_test_final, type = "response") > 0.5, 1, 0)
cat("Predictions complete!\n")

# Save predictions to CSV
output <- data.frame(PassengerId = test$PassengerId, Survived = final_pred)
write.csv(output, "src/data/predictions_r.csv", row.names = FALSE)
cat("Saved predictions to src/data/predictions_r.csv\n")

cat("\nScript complete — Titanic model (R version) finished successfully.\n")
