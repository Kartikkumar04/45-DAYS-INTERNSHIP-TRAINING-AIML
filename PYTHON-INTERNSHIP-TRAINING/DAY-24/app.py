import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Create synthetic dataset
np.random.seed(42)
num_samples = 1000

data = {
    'Bedrooms': np.random.randint(1, 6, size=num_samples),
    'Bathrooms': np.random.randint(1, 4, size=num_samples),
    'SquareFootage': np.random.randint(800, 4000, size=num_samples),
    'Age': np.random.randint(0, 50, size=num_samples),
    'Neighborhood': np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_samples),
    'Garage': np.random.choice(['Yes', 'No'], size=num_samples),
    'SwimmingPool': np.random.choice(['Yes', 'No'], size=num_samples)
}

# Create price based on features with some noise
base_price = (data['Bedrooms'] * 20000 + 
              data['Bathrooms'] * 15000 + 
              data['SquareFootage'] * 100 + 
              (50 - data['Age']) * 1000)

# Adjust for neighborhood
neighborhood_factor = np.where(data['Neighborhood'] == 'Urban', 1.3, 
                              np.where(data['Neighborhood'] == 'Suburban', 1.1, 1.0))
base_price = base_price * neighborhood_factor

# Adjust for garage and pool
base_price = np.where(data['Garage'] == 'Yes', base_price + 15000, base_price)
base_price = np.where(data['SwimmingPool'] == 'Yes', base_price + 25000, base_price)

# Add some random noise
data['Price'] = base_price + np.random.normal(0, 20000, size=num_samples)

df = pd.DataFrame(data)

print(df.head())
print("\nDataset info:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Encode categorical features
label_encoders = {}
categorical_cols = ['Neighborhood', 'Garage', 'SwimmingPool']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

    # Histograms for numerical features
numerical_cols = ['Bedrooms', 'Bathrooms', 'SquareFootage', 'Age', 'Price']
df[numerical_cols].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Scatter plots of price vs numerical features
for col in numerical_cols[:-1]:  # exclude Price itself
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[col], y=df['Price'])
    plt.title(f'Price vs {col}')
    plt.show()

# Boxplots for categorical features
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col], y=df['Price'])
    plt.title(f'Price distribution by {col}')
    plt.show()

    X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)

# Initialize and train the model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"{model_name} Evaluation:")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted Prices ({model_name})')
    plt.show()
    
    return mae, rmse

# Evaluate both models
mae_lr, rmse_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")
mae_dt, rmse_dt = evaluate_model(y_test, y_pred_dt, "Decision Tree")

# For Linear Regression
print("Linear Regression Coefficients:")
for feature, coef in zip(X.columns, lr.coef_):
    print(f"{feature}: {coef:.2f}")

# For Decision Tree
print("\nDecision Tree Feature Importances:")
for feature, importance in zip(X.columns, dt.feature_importances_):
    print(f"{feature}: {importance:.2f}")

# Plot feature importance for Decision Tree
plt.figure(figsize=(10, 6))
sns.barplot(x=dt.feature_importances_, y=X.columns)
plt.title('Decision Tree Feature Importance')
plt.show()

print("\nModel Comparison:")
print(f"Linear Regression MAE: ${mae_lr:,.2f}, RMSE: ${rmse_lr:,.2f}")
print(f"Decision Tree MAE: ${mae_dt:,.2f}, RMSE: ${rmse_dt:,.2f}")

if mae_lr < mae_dt:
    print("\nLinear Regression performs better on MAE.")
else:
    print("\nDecision Tree performs better on MAE.")

if rmse_lr < rmse_dt:
    print("Linear Regression performs better on RMSE.")
else:
    print("Decision Tree performs better on RMSE.")