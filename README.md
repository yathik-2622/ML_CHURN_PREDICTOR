# Telco Customer Churn Prediction – Mini ML Project (Student Setup Guide)
# Refer "train_churn_model.ipynb" this code for Completing the Project

# 📌 Project Overview

This project is developed as part of the ML With Python (October Batch – 2025) training program.
The objective of the project is to build a Customer Churn Prediction System using Machine Learning techniques.

Customer churn refers to when a customer stops using a company’s service.
By analyzing historical customer data, we predict whether a customer will churn (Yes/No).

This project includes:

```
Data preprocessing
Feature engineering
Model training
Evaluation
Saving the model
```

# Creating a Streamlit UI for end-user predictions

🛠️ Technologies Used
```
## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellowgreen)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-lightblue)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Classifier-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-brightgreen)
![Pickle](https://img.shields.io/badge/Pickle-Model%20Saving-yellow)
![Git](https://img.shields.io/badge/Git-Version%20Control-orange)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black)
![VS Code](https://img.shields.io/badge/VS%20Code-Editor-blueviolet)
![CSV](https://img.shields.io/badge/Data-CSV-important)

```
## 1. Download the Project ZIP File
1. Download the ZIP file from Teams.
2. Move it to Desktop.
3. Right-click → Extract All.
4. You will see:

```
Churn_Project/
    ├── Telco-Customer-Churn.csv
    ├── train_churn_model.ipynb
    ├── Project_Submit_ML_Code.py
    ├── app.py
    ├── README.md
```

---

## 2. Install Python
Run:
```
python --version
```

Install Python 3.10 or 3.11 if missing.

---

## 3. Open the Project in VS Code
1. Open VS Code  
2. File → Open Folder  
3. Select the extracted folder  

---

## 4. Create Virtual Environment
```
python -m venv .venv
```

Activate:

Windows:
```
.venv\Scripts\activate
```

Mac:
```
source .venv/bin/activate
```

---

## 5. Install Required Libraries
```
pip install pandas numpy scikit-learn streamlit matplotlib
```

---

## 6. Run Training Notebook (Optional)
Open `Project_Submit_ML_Code.py` → run cells.

This saves:
```
streaming_churn_model.pkl
scaler.pkl
encoder.pkl
```

HOW to run:

```
python filename.py

 (or)

python Project_Submit_ML_Code.py
```
---

## 7. Run Streamlit App
```
streamlit run app.py
```

Opens:
```
http://localhost:8501
```

---

## 8. Common Issues

### Missing module
```
pip install <module>
```

### Streamlit not found
```
pip install streamlit
```

### Feature mismatch  
Retrain the model.

---

## 9. Expected Folder Structure
```
Churn_Project/
    ├── Telco-Customer-Churn.csv
    ├── train_churn_model.ipynb
    ├── Project_Submit_ML_Code.ipynb
    ├── app.py
    ├── streaming_churn_model.pkl
    ├── scaler.pkl
    ├── encoder.pkl
    ├── README.md
    └── .venv/
```

---

## You are ready to begin the project! 🎉
