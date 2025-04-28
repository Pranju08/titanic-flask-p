import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Titanic dataset
data = pd.read_csv('train.csv')

# Check if all expected columns are there
print("Columns:", data.columns.tolist())

# Preprocessing
cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId']
existing_cols = [col for col in cols_to_drop if col in data.columns]
data = data.drop(existing_cols, axis=1)

# Handle missing values correctly
if 'Age' in data.columns:
    data['Age'] = data['Age'].fillna(data['Age'].median())

if 'Embarked' in data.columns:
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Encode categorical features
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})

if 'Embarked' in data.columns:
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'titanic_model.pkl'")
