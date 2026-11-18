#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:23:51 2024

@author: krishsarin
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

from model_utils import load_split, make_preprocessor, parse_data_argument, print_evaluation


def main() -> None:
    args = parse_data_argument("Evaluate Local Outlier Factor on the fraud dataset.")
    X_train, X_test, _, y_test = load_split(args.data)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "local_outlier_factor",
                LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True),
            ),
        ]
    )
    pipeline.fit(X_train)
    predictions = pipeline.predict(X_test)
    predictions = np.where(predictions == 1, 0, 1)
    print_evaluation("Local Outlier Factor", y_test, predictions)


if __name__ == "__main__":
    main()
