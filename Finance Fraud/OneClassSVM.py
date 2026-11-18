#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:18:01 2024

@author: krishsarin
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from model_utils import load_split, make_preprocessor, parse_data_argument, print_evaluation


def main() -> None:
    args = parse_data_argument("Evaluate One-Class SVM on the fraud dataset.")
    X_train, X_test, _, y_test = load_split(args.data)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("one_class_svm", OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")),
        ]
    )
    pipeline.fit(X_train)
    predictions = pipeline.predict(X_test)
    predictions = np.where(predictions == 1, 0, 1)
    print_evaluation("One-Class SVM", y_test, predictions)


if __name__ == "__main__":
    main()
