import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the Dataset
data = pd.read_csv("creditcard.csv")

# Step 2: Analyze Class Distribution
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))

print(f"Dataset Loaded. Fraud Cases: {len(fraud)}, Valid Transactions: {len(valid)}")

# Step 3: Preparing Data Split (80% Train, 20% Test)
X = data.drop(['Class'], axis=1)
Y = data["Class"]
xData = X.values
yData = Y.values

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42
)

# Step 4: Building and Training the Random Forest Model
print("Training Random Forest Classifier...")
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)

# Step 5: Evaluating the Model
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
