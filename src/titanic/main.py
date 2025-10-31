import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


print("Loading Titanic dataset...")
train = pd.read_csv("src/data/train.csv")
test = pd.read_csv("src/data/test.csv")
print("Data loaded successfully!")

print("\nExploring and adjusting data...")
# Encode 'Sex' column (convert text to numeric)
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
print("Converted 'Sex' column to numeric.")

## Impute missing values
for col in ["Age", "Fare"]:
    train[col].fillna(train[col].mean(), inplace=True)
    test[col].fillna(test[col].mean(), inplace=True)
print("Filled missing values in 'Age' and 'Fare' with mean values.")

# Step 3: Select features and target
X = train[["Pclass", "Age", "Sex", "Fare"]].copy()
y = train["Survived"]

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train logistic regression model
print("Training logistic regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Evaluate model
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_acc:.3f}")

# Step 8: Test the model
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Test accuracy: {test_acc:.3f}")

print("Model training complete.")

print("\nPredicting on test set...")

# Prepare test features
X_test_final = test[["Pclass", "Sex", "Age", "Fare"]].copy()

X_test_final = X_test_final[X_train.columns]

# Predict survival
predictions = model.predict(X_test_final)
print("Predictions complete!")

# Save predictions to CSV
output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
output.to_csv("src/data/predictions.csv", index=False)
print("Saved predictions to src/data/predictions.csv")

