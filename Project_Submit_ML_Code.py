# ----------------------------------------------------
# STEP 1 — Import Libraries
# ----------------------------------------------------
# TO DO: Import pandas, numpy, train_test_split, StandardScaler, LabelEncoder,
#        RandomForestClassifier, accuracy_score, classification_report, pickle
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
# TO DO: Load the CSV file → example: pd.read_csv("file_name.csv")
df = pd.read_csv("___")    
print("Dataset loaded!")
print(df.shape)
print(df.head())


# ----------------------------------------------------
# STEP 3 — Basic Cleaning
# ----------------------------------------------------
# TO DO: Drop a column named "customerID"
df.drop("___", axis=1, inplace=True)

# TO DO: Convert "TotalCharges" column into numeric using errors='coerce'
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="___")

# TO DO: Fill missing values in TotalCharges with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].___())

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

# TO DO: Select features + target column "Churn"
df = df[selected_features + ["___"]]
print("Selected columns:", df.columns.tolist())


# ----------------------------------------------------
# STEP 5 — Identify categorical & numeric columns
# ----------------------------------------------------
# TO DO: Identify categorical columns using select_dtypes(include="object")
cat_cols = df.select_dtypes(include="___").columns.tolist()

# TO DO: Identify numeric columns using int64 & float64
num_cols = df.select_dtypes(include=['___', '___']).columns.tolist()

print("Categorical Columns:", cat_cols)
print("Numeric Columns:", num_cols)


# ----------------------------------------------------
# STEP 6 — Encode categorical columns
# ----------------------------------------------------
encoder = LabelEncoder()

# TO DO: Apply label encoding → encoder.fit_transform()
for col in cat_cols:
    df[col] = encoder.___(df[col])

print("Categorical encoding completed using ONE LabelEncoder.")


# ----------------------------------------------------
# STEP 7 — Train/Test Split
# ----------------------------------------------------
# TO DO: Separate X (all features) and y (target column)
X = df.drop("___", axis=1)
y = df["___"]

# TO DO: Perform train-test split with test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=___, random_state=42
)

print("Train/Test split done.")
print("Train:", X_train.shape, "Test:", X_test.shape)


# ----------------------------------------------------
# STEP 8 — Scale numerical columns
# ----------------------------------------------------
scaler = StandardScaler()

# TO DO: Fit & transform X_train numerical columns
X_train[num_cols] = scaler.___(___)

# TO DO: Only transform X_test numerical columns
X_test[num_cols] = scaler.___(___)

print("Scaling done on:", num_cols)


# ----------------------------------------------------
# STEP 9 — Train Model
# ----------------------------------------------------
model = RandomForestClassifier()

# TO DO: Train model → model.fit()
model.___(X_train, y_train)

print("Model trained successfully!")


# ----------------------------------------------------
# STEP 10 — Evaluate
# ----------------------------------------------------
# TO DO: Predict using model.predict()
y_pred = model.___(X_test)

# TO DO: Print accuracy_score and classification_report
print("\nAccuracy:", accuracy_score(___, ___))
print("\nReport:\n", classification_report(___, ___))


# ----------------------------------------------------
# STEP 11 — Save model, scaler, encoder
# ----------------------------------------------------
# TO DO: Save model as streaming_churn_model.pkl
pickle.dump(model, open("___", "wb"))

# TO DO: Save scaler.pkl
pickle.dump(scaler, open("___", "wb"))

# TO DO: Save encoder.pkl
pickle.dump(encoder, open("___", "wb"))

print("Model, scaler, encoder saved!")
