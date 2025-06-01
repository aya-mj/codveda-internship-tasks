import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv('Datasets/churn-bigml-80.csv')

categorical_cols = ['State', 'Area code', 'International plan', 'Voice mail plan']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

le = LabelEncoder()
data_encoded['Churn'] = le.fit_transform(data_encoded['Churn']) 

X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

models = {
    'Logistic Regression': (logreg_model, X_test_scaled), 
    'Random Forest': (rf_model, X_test), 
    'SVM': (svm_model, X_test_scaled)
}
results = {}

for name, (model, X_test_data) in models.items():
    y_pred = model.predict(X_test_data)
    y_proba = model.predict_proba(X_test_data)[:, 1]  # Get probabilities for positive class
    
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    results[name] = y_proba

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for name, y_proba in results.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(True)
plt.show()