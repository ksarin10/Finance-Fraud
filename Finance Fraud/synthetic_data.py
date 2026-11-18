import pandas as pd
import random
from faker import Faker
import json
from pathlib import Path
import argparse

fake = Faker()
Faker.seed(42)
random.seed(42)

DEFAULT_MAPPING_PATH = Path(__file__).with_name('ip_to_zip.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate illustrative synthetic transactions.')
    parser.add_argument('--mapping', type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument('--output', type=Path, default=Path('synthetic_fraud_data.csv'))
    return parser.parse_args()


args = parse_args()

with args.mapping.open() as f:
    ip_to_zip_mappings = json.load(f)


def generate_transaction(fraud=False, multiple_fraud=False):
    name = 'John Smith'
    email = 'john.smith@example.com'
    transaction_amount = round(random.uniform(1, 20000), 2)
    time = fake.time()
    zip_code = '10001'  # Majority ZIP code for John Smith
    ip_mapping = random.choice(ip_to_zip_mappings)
    ip_address = ip_mapping['ip']
    transaction_type = random.choice(['Online', 'In-Store', 'Mobile'])

    if fraud:
        fraud_types = ['zip_ip_mismatch', 'suspicious_time', 'email_change']
        selected_fraud_types = random.sample(fraud_types, k=2 if multiple_fraud else 1)
        for fraud_type in selected_fraud_types:
            if fraud_type == 'zip_ip_mismatch':
                ip_mapping = random.choice(ip_to_zip_mappings)
                zip_code = ip_mapping['postal']
                ip_address = fake.ipv4()
            elif fraud_type == 'suspicious_time':
                time = fake.date_time_this_year(before_now=True, after_now=False, tzinfo=None).replace(hour=random.choice([1, 2, 3, 4]))
            elif fraud_type == 'email_change':
                email = fake.email()
        
    return {
        'Name': name,
        'Email': email,
        'Transaction Amount': transaction_amount,
        'Time': time,
        'ZIP': zip_code,
        'IP': ip_address,
        'Transaction Type': transaction_type
    }


data = []
for _ in range(13050):  
    data.append(generate_transaction(fraud=False))
for _ in range(1350):  
    data.append(generate_transaction(fraud=True, multiple_fraud=False))
for _ in range(450):  
    data.append(generate_transaction(fraud=True, multiple_fraud=True))

df = pd.DataFrame(data)

# Save to CSV
df.to_csv(args.output, index=False)

# Display the first few rows
print(df.head())
print(f"Wrote {len(df):,} rows to {args.output}")
