rm(list=ls())
#libraries
library(caret)
library(rpart)  #Decision Trees
library(pROC)   #ROC curve
library(plyr)   
library(ggplot2)  #plotting
library(rpart.plot)


#Load the dataset
data <- read_csv("C:/Users/Desktop/Healthcare-Diabetes.csv")

#Train-Test Split
set.seed(123)
train_indices <- sample(nrow(data), 0.8 * nrow(data)) # 80% for training
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

#Scaling
min_max_scale <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

columns_to_scale <- setdiff(names(train_data), c("Id", "Outcome"))
train_data_scaled <- as.data.frame(lapply(train_data[columns_to_scale], min_max_scale))
test_data_scaled <- as.data.frame(lapply(test_data[columns_to_scale], min_max_scale))

#(PCA)
pca_train <- prcomp(train_data_scaled)
train_data_pca <- as.data.frame(pca_train$x)

#Model Training and Evaluation (Decision Trees - CART)
cart_model <- rpart(Outcome ~ ., data = train_data, method = "class")
cart_pred <- predict(cart_model, test_data, type = "class")

# Step 5: Performance Evaluation
# Confusion Matrix for Decision Trees (CART)
print("Confusion Matrix for Decision Trees (CART):")
confusion_matrix_cart <- table(cart_pred, test_data$Outcome)
print(confusion_matrix_cart)

#accuracy
accuracy_cart <- sum(diag(confusion_matrix_cart)) / sum(confusion_matrix_cart)
print("Accuracy of Decision Trees (CART):")
print(accuracy_cart)

#ROC Curve and AUC for Decision Trees (CART)
roc_cart <- roc(test_data$Outcome, as.numeric(cart_pred))
auc_cart <- auc(roc_cart)

#ROC Curve
plot(roc_cart, main = "ROC Curve for Decision Trees (CART)")
#probabilities for positive class
cart_prob <- predict(cart_model, test_data)[, "1"]

#Sort predictions and actual outcomes by predicted probabilities
sorted_data <- data.frame(Actual = test_data$Outcome, Predicted_Prob = cart_prob)
sorted_data <- sorted_data[order(-sorted_data$Predicted_Prob), ]

#cumulative gains
cumulative_gains <- cumsum(sorted_data$Actual) / sum(sorted_data$Actual)

#Plot Cumulative Gains Chart
plot(1:length(cumulative_gains), cumulative_gains, type = "l", 
     xlab = "Percentage of Population", ylab = "Cumulative Gains",
     main = "Cumulative Gains Chart for Decision Trees (CART)")




#decline wise

# Calculate the total number of positive outcomes
total_positives <- sum(test_data$Outcome)

# Calculate the total number of observations
total_obs <- length(test_data$Outcome)

# Calculate the random response rate
random_response_rate <- total_positives / total_obs

# Calculate the sorted response rates
sorted_data$response_rate <- cumsum(sorted_data$Actual) / (1:length(sorted_data$Actual))

# Calculate the decline-wise lift
decline_lift <- sorted_data$response_rate / random_response_rate

#Plot Decline-Wise Lift Chart
plot(1:length(decline_lift), decline_lift, type = "l", 
     xlab = "Percentage of Population", ylab = "Decline-Wise Lift",
     main = "Decline-Wise Lift Chart for Decision Trees (CART)")


#cumilative lift chart
#total number of positive outcomes
total_positives <- sum(test_data$Outcome)

#total number of observations
total_obs <- length(test_data$Outcome)
random_response_rate <- total_positives / total_obs

#sorted response rates
sorted_data$response_rate <- cumsum(sorted_data$Actual) / (1:length(sorted_data$Actual))

#the expected random cumulative gains
expected_cumulative_gains <- (1:length(sorted_data$response_rate)) * random_response_rate

#Plot Cumulative Lift Chart
plot(1:length(sorted_data$response_rate), sorted_data$response_rate, type = "l", 
     xlab = "Percentage of Population", ylab = "Cumulative Gains",
     main = "Cumulative Lift Chart for Decision Trees (CART)")


lines(1:length(sorted_data$response_rate), expected_cumulative_gains, col = "red", lty = 2)

legend("bottomright", legend = c("Cumulative Gains", "Random Selection"),
       col = c("black", "red"), lty = c(1, 2))





# Plot the decision tree
prp(cart_model, extra = 1, main = "Decision Tree (CART)")


#Outcome variable as a factor
train_data$Outcome <- as.factor(train_data$Outcome)
test_data$Outcome <- as.factor(test_data$Outcome)

#Train Random Forest model
rf_model <- randomForest(Outcome ~ ., data = train_data, ntree = 100)

#Predict using Random Forest model
rf_pred <- predict(rf_model, test_data)

#Calculate confusion matrix for Random Forest
rf_confusion_matrix <- confusionMatrix(rf_pred, test_data$Outcome)

#Print confusion matrix for Random Forest
cat("Confusion Matrix for Random Forest:\n")
print(rf_confusion_matrix)

#Calculate accuracy for Random Forest
accuracy_rf <- rf_confusion_matrix$overall['Accuracy']

#Print accuracy of Random Forest
print("Accuracy of Random Forest:")
print(accuracy_rf)








