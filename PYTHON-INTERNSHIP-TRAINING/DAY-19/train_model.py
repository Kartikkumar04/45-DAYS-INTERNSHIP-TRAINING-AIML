import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('D:/45-DAYS-INTERNSHIP-TRAINING-AIML/PYTHON-INTERNSHIP-TRAINING/DAY-19/Car_Price_Prediction.csv')


# Encode categorical columns
le_make = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_trans = LabelEncoder()

df['Make'] = le_make.fit_transform(df['Make'])
df['Model'] = le_model.fit_transform(df['Model'])
df['Fuel Type'] = le_fuel.fit_transform(df['Fuel Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# Define feature columns and target
X = df[['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']]
y = (df['Price'] > df['Price'].median()).astype(int)  # Binary classification

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump((le_make, le_model, le_fuel, le_trans), 'encoders.pkl')
