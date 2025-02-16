import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score, roc_curve
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Load the dataset
url = "https://github.com/abdelDebug/Heart-Disease-Data/blob/main/heart_disease.csv?raw=true"
data = pd.read_csv(url)

# Data Preprocessing

data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
data['cp'] = data['cp'].map({'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3,
                             'asymptomatic': 4})
data['restecg'] = data['restecg'].map({'lv hypertrophy': 0, 'normal': 1, 'st-t abnormality': 2})
data['slope'] = data['slope'].map({'upsloping': 1, 'flat': 2, 'downsloping': 3})
data['thal'] = data['thal'].map({'normal': 0, 'fixed defect': 1, 'reversable defect': 2})
data.rename(columns={'num': 'target'}, inplace=True)


data.drop('dataset', axis=1, inplace=True)
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

# Checking for missing values
print(data.isnull().sum())
# print(data.info())

# Separating numeric and boolean columns
boolean_cols = ['fbs', 'exang']
numeric_cols = data.select_dtypes(include=['number']).columns

# Handle boolean columns separately using mode imputation
for col in boolean_cols:
    mode_value = data[col].mode()[0]
    data[col] = data[col].fillna(mode_value)
    data[col] = data[col].round().astype(int)

# Handling missing values in numeric columns (Mean has 2% more accuracy than KNN Imputer)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# knn_imputer = KNNImputer(n_neighbors=5)
# data[numeric_cols] = knn_imputer.fit_transform(data[numeric_cols])

# Separate features and target variable
X = data.drop('target', axis=1)  # Features
y = data['target']               # Target variable (heart disease presence)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model
rf = RandomForestClassifier(
    n_estimators=50,       # Reduce the number of trees
    max_depth=5,           # Limit the depth of trees
    min_samples_split=4,   # Minimum samples required to split
    min_samples_leaf=2,    # Minimum samples in leaf nodes
    random_state=42
)

# rf.fit(X_train, y_train)
# Train the model
rf.fit(X_train_scaled, y_train)

# Make predictions
# y_pred = rf.predict(X_test)
y_pred = rf.predict(X_test_scaled)
# y_pred_prob = rf.predict_proba(X_test)[:, 1]  # For ROC curve, need probabilities
y_pred_prob = rf.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
# Classification report for Precision, Recall, F1-Score
print("Classification Report:")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# AUC-ROC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {roc_auc}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
