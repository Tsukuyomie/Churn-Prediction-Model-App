import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r"archive\dataset.csv")

# Drop customerID if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Remove blank spaces in TotalCharges and convert to float
df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values in selected columns
df = df.dropna(subset=['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract', 'Churn'])

# Encode Contract type
df['Contract'] = LabelEncoder().fit_transform(df['Contract'].astype(str))

# Encode Churn
df['Churn'] = LabelEncoder().fit_transform(df['Churn'])

# Select top 4 features
top_features = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract']
X = df[top_features]
y = df['Churn']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model + features
joblib.dump(model, "top4_churn_model.pkl")
joblib.dump(top_features, "top4_features.pkl")
