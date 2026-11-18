import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope

from model_utils import load_split, make_preprocessor, parse_data_argument, print_evaluation


def main() -> None:
    args = parse_data_argument("Evaluate Elliptic Envelope on the fraud dataset.")
    X_train, X_test, _, y_test = load_split(args.data)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("elliptic_envelope", EllipticEnvelope(contamination=0.1, random_state=42)),
        ]
    )
    pipeline.fit(X_train)
    predictions = pipeline.predict(X_test)
    predictions = np.where(predictions == 1, 0, 1)
    print_evaluation("Elliptic Envelope", y_test, predictions)


if __name__ == "__main__":
    main()
