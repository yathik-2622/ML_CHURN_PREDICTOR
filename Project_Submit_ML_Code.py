# ----------------------------------------------------
# STEP 1 — Import Libraries
# ----------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Libraries imported successfully!")


# ----------------------------------------------------
# STEP 2 — Load Dataset
# ----------------------------------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")
print("Dataset loaded!")
print(df.shape)
print(df.head())


# ----------------------------------------------------
# STEP 3 — Basic Cleaning
# ----------------------------------------------------
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges → numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("Cleaning done.")


# ----------------------------------------------------
# STEP 4 — Select ONLY 8 FEATURES
# ----------------------------------------------------
selected_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport"
]

df = df[selected_features + ["Churn"]]
print("Selected columns:", df.columns.tolist())


# ----------------------------------------------------
# STEP 5 — Identify categorical & numeric columns
# ----------------------------------------------------
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

print("Categorical Columns:", cat_cols)
print("Numeric Columns:", num_cols)


# ----------------------------------------------------
# STEP 6 — Encode categorical columns
# ----------------------------------------------------
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("Categorical encoding completed using ONE LabelEncoder.")


# ----------------------------------------------------
# STEP 7 — Train/Test Split
# ----------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test split done.")
print("Train:", X_train.shape, "Test:", X_test.shape)


# ----------------------------------------------------
# STEP 8 — Scale numerical columns
# ----------------------------------------------------
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("Scaling done on:", num_cols)


# ----------------------------------------------------
# STEP 9 — Train Model
# ----------------------------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully!")


# ----------------------------------------------------
# STEP 10 — Evaluate
# ----------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))



# ----------------------------------------------------
# STEP 11 — Save model, scaler, encoder
# ----------------------------------------------------
pickle.dump(model, open("streaming_churn_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model, scaler, encoder saved!")
