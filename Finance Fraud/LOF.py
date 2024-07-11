#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:23:51 2024

@author: krishsarin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/krishsarin/Downloads/Finance Fraud/creditcardcsvpresent.csv')

print(df.head())

numerical_features = ['Average Amount/transaction/day', 'Transaction_amount', 'Total Number of declines/day',
                      'Daily_chargeback_avg_amt', '6_month_avg_chbk_amt', '6-month_chbk_freq']
categorical_features = ['isForeignTransaction', 'isHighRiskCountry']


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

# Pipeline for Local Outlier Factor
pipeline_lof = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('local_outlier_factor', LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True))
])

# Fit the model
pipeline_lof.fit(X_train)

# Transform the test set
X_test_transformed = pipeline_lof['preprocessor'].transform(X_test)
anomaly_scores_lof = pipeline_lof['local_outlier_factor'].negative_outlier_factor_
anomalies_lof = pipeline_lof['local_outlier_factor'].predict(X_test_transformed)

# Convert predictions to binary: 1 for anomaly, 0 for normal
anomalies_lof = np.where(anomalies_lof == 1, 0, 1)

# Create a DataFrame for test results
df_test_results_lof = X_test.copy()
df_test_results_lof['Anomaly Score'] = anomaly_scores_lof
df_test_results_lof['Predicted Anomaly'] = anomalies_lof
df_test_results_lof['Actual Anomaly'] = y_test.values

# Evaluate the model
print("Local Outlier Factor")
print(classification_report(y_test, anomalies_lof))
print(confusion_matrix(y_test, anomalies_lof))
