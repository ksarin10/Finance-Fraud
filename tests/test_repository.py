import csv
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Finance Fraud"


class RepositoryDataTests(unittest.TestCase):
    def test_dataset_contract(self):
        with (PROJECT / "creditcardcsvpresent.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))

        self.assertEqual(len(rows), 3075)
        self.assertEqual(Counter(row["isFradulent"] for row in rows), {"N": 2627, "Y": 448})
        self.assertTrue(all(row["Transaction date"] == "" for row in rows))

    def test_ip_mapping_contract(self):
        with (PROJECT / "ip_to_zip.json").open() as source:
            mappings = json.load(source)

        self.assertGreater(len(mappings), 0)
        self.assertTrue(all(set(item) == {"ip", "postal"} for item in mappings))


if __name__ == "__main__":
    unittest.main()
