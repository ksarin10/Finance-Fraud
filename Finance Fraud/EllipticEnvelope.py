import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/krishsarin/Downloads/Finance Fraud/creditcardcsvpresent.csv')

print(df.head())


numerical_features = ['Average Amount/transaction/day', 'Transaction_amount', 'Total Number of declines/day',
                      'Daily_chargeback_avg_amt', '6_month_avg_chbk_amt', '6-month_chbk_freq']
categorical_features = ['isForeignTransaction', 'isHighRiskCountry']

# Preprocessing transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define features and target
X = df.drop(columns=['Merchant_id', 'Transaction date', 'isFradulent', 'Is declined'])
y = df['isFradulent'].map({"N": 0, "Y": 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Pipeline for Elliptic Envelope
pipeline_elliptic = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('elliptic_envelope', EllipticEnvelope(contamination=0.1))
])

# Fit the model
pipeline_elliptic.fit(X_train)

# Transform the test set
X_test_transformed = pipeline_elliptic['preprocessor'].transform(X_test)
anomaly_scores_elliptic = pipeline_elliptic['elliptic_envelope'].decision_function(X_test_transformed)
anomalies_elliptic = pipeline_elliptic['elliptic_envelope'].predict(X_test_transformed)

# Convert predictions to binary: 1 for anomaly, 0 for normal
anomalies_elliptic = np.where(anomalies_elliptic == 1, 0, 1)

# Create a DataFrame for test results
df_test_results_elliptic = X_test.copy()
df_test_results_elliptic['Anomaly Score'] = anomaly_scores_elliptic
df_test_results_elliptic['Predicted Anomaly'] = anomalies_elliptic
df_test_results_elliptic['Actual Anomaly'] = y_test.values

# Evaluate the model
print("Elliptic Envelope")
print(classification_report(y_test, anomalies_elliptic))
print(confusion_matrix(y_test, anomalies_elliptic))
