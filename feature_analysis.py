import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance


df = pd.read_csv(r"archive\dataset.csv")


if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

if df["Churn"].dtype == "object":
    df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by="Importance", ascending=False)
top_n = 10

sns.barplot(data=feature_importance_df.head(top_n), x="Importance", y="Feature", palette="viridis")
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

top_4_features = feature_importance_df.head(4)["Feature"].tolist()
print("Top 4 Features for Churn Prediction:", top_4_features)
