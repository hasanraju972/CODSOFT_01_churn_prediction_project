import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Load the dataset
file_path = 'Churn_Modelling.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Encode categorical variables
label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Define features and target variable
X = data.drop(columns=['Exited'])
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)

# Train and evaluate Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)

# Train and evaluate Gradient Boosting
grad_boost = GradientBoostingClassifier(random_state=42)
grad_boost.fit(X_train, y_train)
y_pred_grad_boost = grad_boost.predict(X_test)
grad_boost_accuracy = accuracy_score(y_test, y_pred_grad_boost)

# Generate classification reports and confusion matrices
log_reg_report = classification_report(y_test, y_pred_log_reg)
log_reg_conf_matrix = confusion_matrix(y_test, y_pred_log_reg)

random_forest_report = classification_report(y_test, y_pred_random_forest)
random_forest_conf_matrix = confusion_matrix(y_test, y_pred_random_forest)

grad_boost_report = classification_report(y_test, y_pred_grad_boost)
grad_boost_conf_matrix = confusion_matrix(y_test, y_pred_grad_boost)

# Print results
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Logistic Regression Classification Report:\n", log_reg_report)
print("Logistic Regression Confusion Matrix:\n", log_reg_conf_matrix)

print("Random Forest Accuracy:", random_forest_accuracy)
print("Random Forest Classification Report:\n", random_forest_report)
print("Random Forest Confusion Matrix:\n", random_forest_conf_matrix)

print("Gradient Boosting Accuracy:", grad_boost_accuracy)
print("Gradient Boosting Classification Report:\n", grad_boost_report)
print("Gradient Boosting Confusion Matrix:\n", grad_boost_conf_matrix)

# Visualization functions
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_roc_curve(y_test, y_probs, title):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()

def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title(title)
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(log_reg_conf_matrix, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(random_forest_conf_matrix, 'Random Forest Confusion Matrix')
plot_confusion_matrix(grad_boost_conf_matrix, 'Gradient Boosting Confusion Matrix')

# Plot ROC curves
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
random_forest_probs = random_forest.predict_proba(X_test)[:, 1]
grad_boost_probs = grad_boost.predict_proba(X_test)[:, 1]

plot_roc_curve(y_test, log_reg_probs, 'Logistic Regression ROC Curve')
plot_roc_curve(y_test, random_forest_probs, 'Random Forest ROC Curve')
plot_roc_curve(y_test, grad_boost_probs, 'Gradient Boosting ROC Curve')

# Plot feature importance for Random Forest and Gradient Boosting
plot_feature_importance(random_forest, X.columns, 'Random Forest Feature Importance')
plot_feature_importance(grad_boost, X.columns, 'Gradient Boosting Feature Importance')
