import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# Drop columns not needed for analysis
data_train = data_train.drop(columns=['Cabin', 'Ticket', 'Name', 'Embarked'])
data_test = data_test.drop(columns=['Cabin', 'Ticket', 'Name', 'Embarked'])

# Fill missing values in 'Age' and 'Fare' columns (use median as a simple imputation strategy)
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].median())

# Encode categorical columns
label_enc = LabelEncoder()
data_train['Sex'] = label_enc.fit_transform(data_train['Sex'])
data_test['Sex'] = label_enc.transform(data_test['Sex'])

# Define feature set and target variable
X_train = data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y_train = data_train['Survived']

# Train the model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
X_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y_test_pred = random_forest_model.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': data_test['PassengerId'],  # Use the PassengerId from the test set
    'Survived': y_test_pred
})

submission.to_csv('submission_random_forest.csv', index=False)
