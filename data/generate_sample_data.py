"""
Sample Dataset Generator
Generates realistic synthetic transaction data for testing.
"""

import json
import uuid
import time
import random
import csv
from pathlib import Path

random.seed(42)

COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "DE", "FR", "AU", "NG", "RU", "CN", "BR"]
MERCHANT_CATS = ["retail", "grocery", "restaurant", "travel", "electronics", "healthcare",
                 "entertainment", "crypto", "gambling", "wire_transfer", "gas_station", "online_shopping"]
DEVICES = ["mobile", "desktop", "tablet", "unknown"]
CHANNELS = ["online", "mobile_app", "pos", "atm"]
USER_IDS = [f"user_{i:04d}" for i in range(1, 201)]
DEVICE_IDS = [f"dev_{str(uuid.uuid4())[:8]}" for _ in range(150)]
MERCHANT_IDS = [f"merch_{i:04d}" for i in range(1, 80)]

def gen_ip():
    # Some "risky" IPs
    if random.random() < 0.1:
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def generate_transaction(is_fraud: bool = False) -> dict:
    user_id = random.choice(USER_IDS)
    
    if is_fraud:
        amount = round(random.uniform(1500, 8000), 2)
        country = random.choice(["NG", "RU", "CN", "BR", "IR"])
        merchant_cat = random.choice(["crypto", "wire_transfer", "gambling"])
        device_id = f"dev_new_{str(uuid.uuid4())[:6]}"  # always new device
    else:
        amount = round(random.expovariate(1/150), 2)
        amount = min(max(amount, 1.0), 3000)
        country = random.choice(COUNTRIES)
        merchant_cat = random.choice(MERCHANT_CATS[:9])  # normal categories
        device_id = random.choice(DEVICE_IDS)

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": amount,
        "currency": "USD",
        "merchant_id": random.choice(MERCHANT_IDS),
        "merchant_category": merchant_cat,
        "country": country,
        "ip_address": gen_ip(),
        "device_id": device_id,
        "device_type": random.choice(DEVICES),
        "session_id": str(uuid.uuid4())[:16],
        "channel": random.choice(CHANNELS),
        "timestamp": time.time() - random.uniform(0, 86400 * 7),
        "label": 1 if is_fraud else 0,
    }


def generate_dataset(n_normal: int = 950, n_fraud: int = 50, output_dir: str = "."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    transactions = []
    transactions += [generate_transaction(False) for _ in range(n_normal)]
    transactions += [generate_transaction(True) for _ in range(n_fraud)]
    random.shuffle(transactions)

    # JSON output
    json_path = Path(output_dir) / "sample_transactions.json"
    with open(json_path, "w") as f:
        json.dump(transactions, f, indent=2)
    
    # CSV output
    csv_path = Path(output_dir) / "sample_transactions.csv"
    if transactions:
        fieldnames = transactions[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transactions)
    
    print(f"✅ Generated {len(transactions)} transactions ({n_fraud} fraud)")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    return transactions


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    generate_dataset(950, 50, str(data_dir))
