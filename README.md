#  Customer Churn Prediction Project

##  Overview
This project predicts customer churn using machine learning.  
It involves data cleaning, feature importance analysis, model training, and a Streamlit-based web app for interactive predictions.  
The workflow follows a simple, structured approach — from understanding key drivers of churn to deploying a working model.

---

## Project Structure

### 1. **Feature Importance Analysis – `feature_analysis.py`**
This script identifies the most influential factors that contribute to customer churn.

**Key Steps:**
- Loads and cleans the dataset.
- Encodes categorical variables using `LabelEncoder`.
- Trains a `RandomForestClassifier` to determine which features matter most.
- Visualizes the **Top 10 features** using a Seaborn bar chart (`feature_importance.png`).
- Prints the **Top 4 features** that will be used in the model training stage.

**Output:**
- `feature_importance.png`
- List of top 4 features for churn prediction.

---

### 2. **Model Training – `train_top4_model.py`**
This script trains the final churn prediction model using the **top 4 selected features**.

**Key Steps:**
- Loads and cleans the dataset.
- Focuses on four main features:  
  `MonthlyCharges`, `tenure`, `TotalCharges`, and `Contract`.
- Encodes categorical fields (`Contract`, `Churn`).
- Trains a `RandomForestClassifier` to predict churn.
- Saves the model and feature list for deployment.

**Output Files:**
- `top4_churn_model.pkl` — trained model  
- `top4_features.pkl` — list of selected features  

---

### 3. **Streamlit App – `app.py`**
This is the interactive web interface where users can input customer details and get a churn prediction.

**How it works:**
- Loads the trained model and feature list.
- Accepts user input:  
  - `Monthly Charges`  
  - `Tenure`  
  - `Total Charges`  
  - `Contract Type`
- Encodes inputs and makes a prediction using the saved model.
- Displays:
  - **Color-coded result card** (Red = Churn, Green = No Churn)
  - **Churn probability gauge** using Plotly for visual feedback.

**To run the app:**
```bash
streamlit run app.py
