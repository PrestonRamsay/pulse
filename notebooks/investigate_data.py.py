
import csv
import os
import glob
from collections import Counter, defaultdict

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory(
    title="Select the folder containing the nine monthly MBTA CSV files"
)
root.destroy()

if not folder:
    print("No folder selected.")
    raise SystemExit

files = sorted(glob.glob(os.path.join(folder, "*.csv")))
if not files:
    print("No CSV files found.")
    raise SystemExit

total_by_type = Counter()
missing_by_type = defaultdict(Counter)
key_counts = Counter()
first_value_for_key = {}
conflicting_keys = set()
exact_rows = Counter()
total_rows = 0

key_fields = (
    "service_date", "route_id", "direction_id",
    "stop_id", "scheduled"
)
compare_fields = (
    "actual", "standard_type", "point_type",
    "half_trip_id", "scheduled_headway", "headway"
)

for path in files:
    print(f"Reading {os.path.basename(path)}...")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1
            record_type = (row.get("standard_type") or "").strip() or "(blank)"
            total_by_type[record_type] += 1

            for field, value in row.items():
                if value is None or not value.strip():
                    missing_by_type[record_type][field] += 1

            key = tuple((row.get(field) or "").strip()
                        for field in key_fields)
            comparison = tuple((row.get(field) or "").strip()
                               for field in compare_fields)

            key_counts[key] += 1
            exact_rows[(key, comparison)] += 1

            if key not in first_value_for_key:
                first_value_for_key[key] = comparison
            elif first_value_for_key[key] != comparison:
                conflicting_keys.add(key)

print("\n========== ROW COUNTS BY STANDARD TYPE ==========")
for record_type, count in total_by_type.items():
    print(f"{record_type}: {count:,} rows")

print("\n========== MISSING VALUES BY STANDARD TYPE ==========")
for record_type, total in total_by_type.items():
    print(f"\n{record_type} ({total:,} rows)")
    for field, count in missing_by_type[record_type].most_common():
        if count:
            print(f"  {field}: {count:,} ({count / total:.2%})")

duplicate_keys = sum(1 for count in key_counts.values() if count > 1)
excess_rows = sum(count - 1 for count in key_counts.values() if count > 1)
exact_duplicate_excess = sum(
    count - 1 for count in exact_rows.values() if count > 1
)

print("\n========== DUPLICATE INVESTIGATION ==========")
print(f"Distinct proposed keys: {len(key_counts):,}")
print(f"Keys appearing more than once: {duplicate_keys:,}")
print(f"Excess rows beyond one per key: {excess_rows:,}")
print(f"Keys with differing compared fields: {len(conflicting_keys):,}")
print(f"Exact repeated key + compared-field combinations (excess rows): "
      f"{exact_duplicate_excess:,}")

print("\nExamples of conflicting keys (up to 10):")
shown = 0
for key in conflicting_keys:
    print("\nKey:", dict(zip(key_fields, key)))
    for comparison_key, count in exact_rows.items():
        if comparison_key[0] == key:
            print(f"  Count {count}: "
                  f"{dict(zip(compare_fields, comparison_key[1]))}")
    shown += 1
    if shown >= 10:
        break

print("\nInvestigation complete.")
