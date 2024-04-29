import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("/Users/vishnuvardhan/Documents/Fintech/Project/Project2/loans_full_schema.csv")

# Inspect the Data
print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Display information about the DataFrame, including data types

# Data Cleaning
# Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill missing values
df.dropna(inplace=True)  # Drop any remaining rows with missing values
# Convert object columns to string type
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

# Data Analysis
# Summary statistics for numeric columns
print(df.describe())

# Data Visualization using Horizontal Bar Plots
# Visualize the frequency of each category for object columns
for col in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index)
    plt.title(f'Horizontal Bar Plot of {col}')
    plt.xlabel('Frequency')
    plt.ylabel(col)
    plt.show()

# Data Transformation
# Perform any necessary data transformations
# For example, you can encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)
# Save the cleaned and transformed data


df = pd.read_csv("/Users/vishnuvardhan/Documents/Fintech/Project/Project2/loans_full_schema.csv")
# Data Preprocessing
# Handle missing values and encode categorical variables
# Handle missing values
df[['verification_income_joint', 'annual_income_joint', 'debt_to_income_joint']] = df[
    ['verification_income_joint', 'annual_income_joint', 'debt_to_income_joint']].fillna('NA')
df['months_since_90d_late'] = df['months_since_90d_late'].fillna('not_late').astype(str)
df['months_since_last_delinq'] = df['months_since_last_delinq'].fillna('not_late_delinq').astype(str)
df['months_since_last_credit_inquiry'] = df['months_since_last_credit_inquiry'].fillna('not_credit_inquiry').astype(str)
df['emp_title'] = df['emp_title'].fillna('No Title').astype(str)
df['emp_length'] = df['emp_length'].fillna('No Length').astype(str)
df['debt_to_income'] = df['debt_to_income'].fillna('joint').astype(str)
df.drop('num_accounts_120d_past_due', inplace=True, axis=1)

# Risk category
def categorize_risk(status):
    if status in ['Fully Paid', 'Current']:
        return 0
    elif status in ['In grace period', 'Late(31-120days)', 'Late(16-30days)']:
        return 1
    else:
        return 1

df['Risk_Category'] = df['loan_status'].apply(categorize_risk)
df.drop('loan_status', inplace=True, axis=1)
categorical_variables = list(df.dtypes[df.dtypes == "object"].index)
numeric_variables = list(df.dtypes[df.dtypes != "object"].index)

df[categorical_variables] = df[categorical_variables].astype(str)

# One-hot encoding
enc = OneHotEncoder(sparse=False)
encoded_data = enc.fit_transform(df[categorical_variables])
feature_names = enc.get_feature_names_out(categorical_variables)
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
numerical_df = df[numeric_variables]

# Standard scaling
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_df)
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_df.columns)
df_scaled = pd.concat([encoded_df, scaled_numerical_df], axis=1)
df_scaled['Risk_Category'] = df['Risk_Category']

# Apply Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_clustered = df_scaled.copy()
df_clustered['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize 3D Clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in np.unique(df_clustered['Cluster']):
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
    ax.scatter(cluster_data.iloc[:, -1], cluster_data.iloc[:, -3], cluster_data.iloc[:, -4], label=f'Cluster {cluster}')

ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, -3], kmeans.cluster_centers_[:, -4],
           marker='x', s=300, c='black', label='Centroids')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Clustering using KMeans')
ax.legend()

plt.show()

# Model Training and Evaluation
X = df_scaled.drop(columns='Risk_Category')
y = df_scaled['Risk_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Initialize models
randf_classifier = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
adb_classifier = AdaBoostClassifier(random_state=42)
log_reg = LogisticRegression(class_weight=class_weight_dict)
svm_clf = SVC(class_weight=class_weight_dict, random_state=42)

models = {
    'RandomForest Classifier': randf_classifier,
    'AdaBoost Classifier': adb_classifier,
    'Logistic Regression': log_reg,
    'SVM Classifier': svm_clf
}

best_models = {}

for name, model in models.items():
    params = {}
    if name == 'RandomForest Classifier':
        params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
    elif name == 'AdaBoost Classifier':
        params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}
    elif name == 'Logistic Regression':
        params = {'C': [0.1, 1, 10]}
    elif name == 'SVM Classifier':
        params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_}")
    print("="*50)
    
    # Evaluate best model
    y_pred = best_models[name].predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)
    
    print(f"{name} Metrics:")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

# nural network

nn_model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
_, accuracy = nn_model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {accuracy}")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Under sampling
undersample = RandomUnderSampler(sampling_strategy='majority')
X_resampled, y_resampled = undersample.fit_resample(X_scaled, y)

# Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize models
randf_classifier = RandomForestClassifier(random_state=42)
adb_classifier = AdaBoostClassifier(random_state=42)
log_reg = LogisticRegression()
svm_clf = SVC()

models = {
    'RandomForest Classifier': randf_classifier,
    'AdaBoost Classifier': adb_classifier,
    'Logistic Regression': log_reg,
    'SVM Classifier': svm_clf
}

best_models = {}

for name, model in models.items():
    params = {}
    if name == 'RandomForest Classifier':
        params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
    elif name == 'AdaBoost Classifier':
        params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}
    elif name == 'Logistic Regression':
        params = {'C': [0.1, 1, 10]}
    elif name == 'SVM Classifier':
        params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_}")
    print("="*50)
    
    # Evaluate best model
    y_pred = best_models[name].predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Charged Off', 'Charged Off'], output_dict=True)
    
    print(f"{name} Metrics:")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print("Classification Report:")
    print(report)

# Neural Network Model
nn_model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
_, accuracy = nn_model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {accuracy}")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
    

