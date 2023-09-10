import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report
from preparing_data import PreparingData
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

# Function to perform oversampling
def perform_oversampling(X_train, y_train):
    # Combine X_train and y_train for resampling
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data['FraudFound_P'] == 0]
    minority_class = train_data[train_data['FraudFound_P'] == 1]

    # Upsample minority class
    minority_upsampled = resample(minority_class,replace=True, n_samples=len(majority_class),random_state=42)

    # Combine majority class with upsampled minority class
    upsampled_data = pd.concat([majority_class, minority_upsampled])

    # Separate features and target after resampling
    y_train_resampled = upsampled_data['FraudFound_P']
    X_train_resampled = upsampled_data.drop('FraudFound_P', axis=1)

    return X_train_resampled, y_train_resampled

# Function to perform undersampling
def perform_undersampling(X_train, y_train):
    # Combine X_train and y_train for resampling
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data['FraudFound_P'] == 0]
    minority_class = train_data[train_data['FraudFound_P'] == 1]

    # Downsample majority class
    majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)

    # Combine downsampled majority class with minority class
    downsampled_data = pd.concat([majority_downsampled, minority_class])

    # Separate features and target after resampling
    y_train_resampled = downsampled_data['FraudFound_P']
    X_train_resampled = downsampled_data.drop('FraudFound_P', axis=1)

    return X_train_resampled, y_train_resampled

# Function to train models, print results and classification reports, and return accuracies and classification reports
def train_and_print_results(models, X_train, y_train, X_test, y_test):
    accuracies = []
    print("Model Results:")   
    for model_name, model in models:
        if isinstance(model, KNeighborsClassifier):
            X_test_contiguous = np.ascontiguousarray(X_test) 
            X_train_contiguous = np.ascontiguousarray(X_train)
            model.fit(X_train_contiguous, y_train)
            accuracy = model.score(X_test_contiguous, y_test)
            y_pred = model.predict(X_test_contiguous)
        elif isinstance(model, LGBMClassifier):
            threshold = 0.541
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_prob > threshold).astype(int) 
        else: 
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
             

        classification_rep = classification_report(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"{model_name}:")
        print(f"Accuracy = {accuracy:.4f}")
        print("Classification Report:")
        print(classification_rep)
        print("="*60)
    return accuracies



# Read the data from the file
data = pd.read_csv("insurance_fraud.csv")
data_copy = data

## Chi-Squared Tests
pvals = np.array([])
for col in data.columns:
    contingency = pd.crosstab(data['FraudFound_P'], data[col])
    pvalue = stats.chi2_contingency(contingency).pvalue
    pvals = np.append(pvals, pvalue)
    
chisqtest = pd.DataFrame({'Column': data.columns.tolist(),
                          'P-value': pvals})

print('Chi-Squared Test Result:\n',chisqtest)
print("="*60)

# Initialize the PreparingData class
data_preparer = PreparingData()

# Preprocess the data
data_preparer.preprocess_data(data)

# Split the data into features and target
y = data_preparer.processed_data['FraudFound_P']
X = data_preparer.processed_data.drop('FraudFound_P', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling
X_train_oversampled, y_train_oversampled = perform_oversampling(X_train, y_train)

# Perform undersampling
X_train_undersampled, y_train_undersampled = perform_undersampling(X_train, y_train)

# Initialize models
models = [
    ("Logistic Regression", LogisticRegression(max_iter=10000)),
    ("Naive Bayesian", GaussianNB()),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=10)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=50)),
    ("Random Forest", RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=5)),
    ("Boosting", GradientBoostingClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10)),
]

# Train models and print results for oversampled data
print("Results for Oversampled Data:")
train_and_print_results(models, X_train_oversampled, y_train_oversampled, X_test, y_test)

# Train models and print results for undersampled data
print("Results for Undersampled Data:")
train_and_print_results(models, X_train_undersampled, y_train_undersampled, X_test, y_test)

print("Results including the column AgeOfPolicyHolder")
model = LGBMClassifier(class_weight='balanced', num_iterations=1000, learning_rate=0.0105)
train_and_print_results([('LGBM classifier', model)], X_train, y_train, X_test, y_test)

print("Results without the column AgeOfPolicyHolder")

data_copy = data_copy.drop("AgeOfPolicyHolder", axis=1)
# Preprocess the data
data_preparer.preprocess_data(data_copy)

# Split the data into features and target
y = data_preparer.processed_data['FraudFound_P']
X = data_preparer.processed_data.drop('FraudFound_P', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMClassifier(class_weight='balanced', num_iterations=1000, learning_rate=0.0105)
train_and_print_results([('LGBM classifier', model)], X_train, y_train, X_test, y_test)
