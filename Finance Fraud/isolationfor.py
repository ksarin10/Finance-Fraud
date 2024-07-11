import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/krishsarin/Downloads/Finance Fraud/creditcardcsvpresent.csv')

print(df.head())



numerical_features = ['Average Amount/transaction/day', 'Transaction_amount', 'Total Number of declines/day',
                      'Daily_chargeback_avg_amt', '6_month_avg_chbk_amt', '6-month_chbk_freq']
categorical_features = ['isForeignTransaction', 'isHighRiskCountry']


numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = df.drop(columns=['Merchant_id', 'Transaction date', 'isFradulent', 'Is declined'])  
y = df['isFradulent'].map({"N" : 0, "Y" : 1})  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('isolation_forest', IsolationForest(contamination=0.1, random_state=42))
])

pipeline.fit(X_train)

X_test_transformed = pipeline['preprocessor'].transform(X_test)
anomaly_scores = pipeline['isolation_forest'].decision_function(X_test_transformed)
anomalies = pipeline['isolation_forest'].predict(X_test_transformed)


anomalies = np.where(anomalies == 1, 0, 1)  

df_test_results = X_test.copy()
df_test_results['Anomaly Score'] = anomaly_scores
df_test_results['Predicted Anomaly'] = anomalies
df_test_results['Actual Anomaly'] = y_test.values

print(classification_report(y_test, anomalies))
print(confusion_matrix(y_test, anomalies))


