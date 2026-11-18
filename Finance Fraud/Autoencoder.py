"""Evaluate a dense autoencoder for fraud anomaly detection."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from model_utils import load_split, make_preprocessor, parse_data_argument, print_evaluation


def main() -> None:
    args = parse_data_argument("Evaluate an autoencoder on the fraud dataset.")
    tf.keras.utils.set_random_seed(42)

    X_train, X_test, _, y_test = load_split(args.data)
    preprocessor = make_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    input_dim = X_train_transformed.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(max(1, input_dim // 2), activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.fit(
        X_train_transformed,
        X_train_transformed,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.2,
        verbose=0,
    )

    reconstructed = autoencoder.predict(X_test_transformed, verbose=0)
    reconstruction_error = np.mean(
        np.square(X_test_transformed - reconstructed), axis=1
    )
    threshold = np.percentile(reconstruction_error, 90)
    predictions = (reconstruction_error > threshold).astype(int)
    print(f"Reconstruction-error threshold: {threshold:.6f}")
    print_evaluation("Autoencoder", y_test, predictions)


if __name__ == "__main__":
    main()
