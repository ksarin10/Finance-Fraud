# Comparative Analysis of Machine Learning Algorithms in Finance Fraud

This project compares five anomaly-detection methods on a labeled synthetic credit-card transaction dataset:

- Isolation Forest
- Elliptic Envelope
- One-Class SVM
- Local Outlier Factor (LOF)
- A dense autoencoder

The goal is to compare how well unsupervised anomaly detectors identify fraudulent transactions after a consistent preprocessing and train/test split. This is a research demonstration, not a production fraud-screening system.

## Dataset

`creditcardcsvpresent.csv` contains 3,075 transactions: 2,627 labeled non-fraudulent (`N`) and 448 labeled fraudulent (`Y`). The code retains the source column name `isFradulent`, including its original spelling.

The experiment uses these model features:

| Type | Features |
| --- | --- |
| Numerical | Average amount per transaction per day, transaction amount, daily decline count, average daily chargeback amount, six-month average chargeback amount, and six-month chargeback frequency |
| Categorical | Foreign transaction indicator and high-risk country indicator |

Merchant ID, transaction date, decline status, and the fraud label are excluded from the model inputs. Numerical features are standardized, and categorical features are one-hot encoded. The scripts use a fixed 70/30 stratified split (`random_state=42`), producing a 923-row test set with 789 non-fraudulent and 134 fraudulent examples.

The transaction-date column is empty in the included dataset. The dataset is synthetic and small, so the reported results should not be treated as estimates of real-world fraud-detection performance.

## Reported results

The accompanying research poster reported the following test-set results:

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 |
| --- | ---: | ---: | ---: | ---: |
| Isolation Forest | 0.91 | 0.82 | 0.53 | 0.64 |
| Elliptic Envelope | 0.93 | 0.87 | 0.57 | 0.69 |
| One-Class SVM | 0.88 | 0.61 | 0.41 | 0.49 |
| Local Outlier Factor | 0.82 | 0.35 | 0.27 | 0.31 |
| Autoencoder | 0.90 | 0.74 | 0.51 | 0.61 |

Elliptic Envelope performed best in this experiment. That result depends on this dataset, preprocessing, split, and hyperparameters. It does not establish that Elliptic Envelope is generally the best fraud model.

### Reproduction note

The scripts were validated on Python 3.13 with scikit-learn 1.9.1 and TensorFlow 2.21.0. Isolation Forest, One-Class SVM, and LOF reproduced the poster's rounded metrics. Elliptic Envelope produced 0.91 accuracy and the autoencoder produced 0.92 accuracy in that environment. Robust covariance estimation and neural-network training can vary across library versions and platforms, so the table above records the poster's reported experiment rather than guaranteeing identical results on every supported setup.

## Setup

Python 3.10 or 3.11 is recommended for the broadest TensorFlow compatibility.

```bash
git clone https://github.com/ksarin10/Finance-Fraud.git
cd "Finance-Fraud/Finance Fraud"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The base requirements run the four scikit-learn models and the data generator. To run the autoencoder, install its additional requirements:

```bash
python -m pip install -r requirements-autoencoder.txt
```

TensorFlow installation varies by Python version and operating system. If installation fails, follow the [official TensorFlow installation guide](https://www.tensorflow.org/install).

## Run the models

Each script defaults to the CSV stored beside it, so it can be launched from the repository root or from the project directory.

```bash
python "Finance Fraud/isolationfor.py"
python "Finance Fraud/EllipticEnvelope.py"
python "Finance Fraud/OneClassSVM.py"
python "Finance Fraud/LOF.py"
python "Finance Fraud/Autoencoder.py"
```

To use another compatible dataset:

```bash
python "Finance Fraud/isolationfor.py" --data /path/to/transactions.csv
```

Each model prints a scikit-learn classification report and confusion matrix. In the output, class `0` means non-fraudulent and class `1` means fraudulent.

## Synthetic transaction generator

`synthetic_data.py` is a separate illustrative generator. It creates person-level transaction fields and does not produce the same schema as `creditcardcsvpresent.csv`, so its output cannot be passed directly to the model scripts.

```bash
cd "Finance Fraud"
python synthetic_data.py --output synthetic_fraud_data.csv
```

## Project files

| File | Purpose |
| --- | --- |
| `model_utils.py` | Shared data validation, preprocessing, split, and reporting |
| `isolationfor.py` | Isolation Forest experiment |
| `EllipticEnvelope.py` | Elliptic Envelope experiment |
| `OneClassSVM.py` | One-Class SVM experiment |
| `LOF.py` | Local Outlier Factor experiment |
| `Autoencoder.py` | TensorFlow/Keras autoencoder experiment |
| `requirements-autoencoder.txt` | Additional dependency for the autoencoder |
| `creditcardcsvpresent.csv` | Included labeled synthetic dataset |
| `synthetic_data.py` | Separate illustrative transaction generator |
| `ip_to_zip.json` | Sample mappings used by the generator |

## Limitations and responsible use

- The included data is synthetic and may not represent current payment behavior or adversarial fraud patterns.
- The scripts train the unsupervised detectors on the full training split, which includes both labels. Labels are used only for stratification and evaluation, not model fitting.
- A fixed contamination rate or percentile threshold assumes a roughly known anomaly rate.
- Accuracy can hide poor fraud recall when fraudulent examples are rare. Review precision, recall, F1, and the confusion matrix together.
- A deployed system would need temporal validation, leakage checks, probability or score calibration, monitoring, human review, and analysis of disparate impact.

Do not use these models to approve, reject, or investigate real transactions without additional validation and appropriate oversight.

## References

1. Bhattacharyya, S., Jha, S., Tharakunnel, K., and Westland, J. C. (2011). “Data mining for credit card fraud: A comparative study.” *Decision Support Systems*, 50(3), 602–613. <https://doi.org/10.1016/j.dss.2010.08.008>
2. Chandradeva, L. S., Jayasooriya, I., and Aponso, A. C. (2019). “Fraud Detection Solution for Monetary Transactions with Autoencoders.” *2019 National Information Technology Conference*, 31–34. <https://doi.org/10.1109/NITC48475.2019.9114519>
3. Panthakkan, A., et al. (2022). “Performance Comparison of Credit Card Fraud Detection System using Machine Learning.” *2022 5th International Conference on Signal Processing and Information Security*, 17–21. <https://doi.org/10.1109/ICSPIS57063.2022.10002517>
4. Vanini, P., Rossi, S., Zvizdic, E., and Domenig, T. (2023). “Online payment fraud: from anomaly detection to risk management.” *Financial Innovation*, 9, 66. <https://doi.org/10.1186/s40854-023-00470-w>
5. Saha, P., et al. (2023). “Comparative Analysis of ML Algorithms for Fraud Detection in Financial Transactions.” *2023 First International Conference on Advances in Electrical, Electronics and Computational Intelligence*, 1–6. <https://doi.org/10.1109/ICAEECI58247.2023.10370930>

## License

No license file is currently included. Copyright law therefore applies by default. Add a license before inviting reuse or outside contributions.
