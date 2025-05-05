# -*- coding: utf-8 -*-
"""
Created on Sat May  3 00:21:43 2025

@author: Chowdhury Arafat Hossain
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

os.chdir("D:\OneDrive - Tulane University\Tulane Uni\Level 1 Term 2\Data Science\Final Exam\Final_Exam")

# Load dataset
df = pd.read_csv("Diabetes_Pred.csv")
df = df.drop(columns=["Unnamed: 0", "X"], errors="ignore")  # drop irrelevant columns

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features and target
X = df_encoded.drop("Diabetes", axis=1)
y = df_encoded["Diabetes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Neural Network": MLPClassifier(max_iter=500),
    "Naive Bayes": GaussianNB()
}

# Evaluate models
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    })

# Display performance
results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
print(results_df)

#2. Visual Comparision by Performance Metrics
# Define the metrics to plot
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

# Create subplots

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    data_sorted = results_df.sort_values(by=metric, ascending=False)
    
    # Use Seaborn deep palette ranked by metric value
    palette = sns.color_palette("deep", len(data_sorted))
    
    ax = axes[i]
    barplot = sns.barplot(x="Model", y=metric, data=data_sorted, palette=palette, ax=ax)
    ax.set_title(f"{metric} by Model", fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.set_xlabel("Ranked Models", fontsize=16)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.tick_params(axis='y', labelsize=12)

    # Add model names inside each bar
    for p, model_name in zip(barplot.patches, data_sorted["Model"]):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.0,
                height / 2,
                model_name,
                ha='center', va='center',
                fontsize=16, rotation=90, color='white', fontweight='bold')

# Hide unused subplot
for j in range(len(metrics_to_plot), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

#Varibale importance and Logistic model Equation
log_reg_model = models["Logistic Regression"]

# Get feature names (assuming X_train is a DataFrame)
features = X_train.columns

# Extract coefficients and intercept
coefficients = log_reg_model.coef_[0]
intercept = log_reg_model.intercept_[0]

# Create DataFrame for plotting
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
    'Importance': np.abs(coefficients)
}).sort_values(by='Importance', ascending=True)

# Plot variable importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=coef_df, palette="crest")
plt.title("Variable Importance - Logistic Regression", fontsize=16)
plt.xlabel("Absolute Coefficient Value", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

###########remodel with selected variables ########################
from sklearn.linear_model import LogisticRegression

# Select features with |coef| > 0.01
selected_mask = np.abs(coefficients) > 0.01
selected_features = features[selected_mask]

print("Selected Features (|coefficient| > 0.01):")
print(selected_features)

# Step 3: Subset training and testing sets
X_train_selected = X_train_scaled[:, selected_mask]
X_test_selected = X_test_scaled[:, selected_mask]

# Step 4: Refit logistic regression on selected features
refit_log_reg = LogisticRegression(max_iter=1000)
refit_log_reg.fit(X_train_selected, y_train)

#comapre Performance between full and feature selcted
log_results = []

# Evaluate original logistic regression
log_reg = models["Logistic Regression"]
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

log_results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba)
})

# Evaluate refit logistic regression with selected features
y_pred = refit_log_reg.predict(X_test_selected)
y_proba = refit_log_reg.predict_proba(X_test_selected)[:, 1]

log_results.append({
    "Model": "Feature Selected Logistic",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba)
})

# Convert to DataFrame for display
results_df = pd.DataFrame(log_results)
print(results_df)
# Step 5: Print the new equation
new_coefficients = refit_log_reg.coef_[0]
intercept = refit_log_reg.intercept_[0]

equation = f"logit(p) = {intercept:.4f}"
for coef, feature in zip(new_coefficients, selected_features):
    sign = " + " if coef >= 0 else " - "
    equation += f"{sign}{abs(coef):.4f} * {feature}"

print("\nUpdated Logistic Regression Equation (|coef| > 0.1):")
print(equation)
#3.	Identify the top-performing model giving reasons
'''Random Forest, Logistic Regression, and Naive Bayes models all achieve the highest accuracy (0.9040), precision (0.904000),
recall (1.000000), and F1 score (0.949580). However, among these, the Random Forest model stands out slightly due to its higher
 ROC AUC (0.514958) compared to Logistic Regression (0.504413) and Naive Bayes (0.503414). However, The AUC value difference between Random
 Forest and Logistic regression is not significantly different (only by 0.01) but the interpretibility of logistic regression is higher than
 The Random Forest. Therefore we would prefer to select Logistic regression as our final model'''

#4.	What are some insights gathered from this project that are of public health relevance (Discuss your results in light of public health)
'''logit(p) = 2.2165 + 0.0361 * Age - 0.0297 * Sex - 0.0314 * Ethnicity - 0.0340 * BMI - 0.0430 * Blood_Pressure_Systolic - 0.0313 * Blood_Pressure_Diastolic 
            + 0.0125 * Cholesterol_Total + 0.0422 * Cholesterol_LDL - 0.0559 * Physical_Activity_Level + 0.0595 * Dietary_Intake_Calories + 0.0358 * Smoking_Status 
            - 0.0241 * Family_History_of_Diabetes - 0.0583 * Previous_Gestational_Diabetes
 
Our Final feature selected model is robust'''
print(results_df)
 
''' From The Analysis We understand that the risk of diabetes increases with the increase of age, Cholestrol LDL, Dietary Intake Calories, and Smoking.
 Total Cholestrol and Cholestrol HDL is not a significant risk factor. Besides, Physical Activity has a protective effect on diabetes. The effect varies
 sex and ethnic group.'''

#5.	What are some caveats and alternatives for the project? 
'''From this model the estimate on Blood pressure (both systolic and diastolic) might be misleading as a continuous variable was passed into the model. 
Surprisngly,The results shows that previous gestational diabetes might have a protective effect on later diabetes. Further modelling appraoch, catagerozing
the variables with medical significance might increase the confidence on the insight from the model.'''
