
import csv
import os
import glob
from collections import Counter
from datetime import datetime

import tkinter as tk
from tkinter import filedialog

# Select the folder containing the nine monthly CSV files.
root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory(
    title="Select the folder containing the monthly MBTA CSV files"
)
root.destroy()

if not folder:
    print("No folder selected.")
    raise SystemExit

files = sorted(glob.glob(os.path.join(folder, "*.csv")))
if not files:
    print("No CSV files found in that folder.")
    raise SystemExit

print(f"Found {len(files)} CSV file(s).")
print("Analyzing every row; this may take a little while.\n")

total_rows = 0
missing = Counter()
value_counts = {
    "point_type": Counter(),
    "standard_type": Counter(),
    "direction_id": Counter(),
    "route_id": Counter(),
}
route_examples = set()
stop_examples = set()
delay_counts = Counter()
delay_values = []
key_counts = Counter()
monthly_rows = Counter()
invalid_timestamps = 0
actual_missing = 0
scheduled_missing = 0
sample_rows = []

def parse_time(value):
    if not value:
        return None
    try:
        # These timestamps are time-of-day values with a 1900 date.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

for path in files:
    filename = os.path.basename(path)
    file_rows = 0
    print(f"Reading {filename}...")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        for row in reader:
            total_rows += 1
            file_rows += 1

            if len(sample_rows) < 5:
                sample_rows.append((filename, row.copy()))

            for column in columns:
                value = row.get(column)
                if value is None or not value.strip():
                    missing[column] += 1

            for column in value_counts:
                value_counts[column][(row.get(column) or "").strip()] += 1

            route = (row.get("route_id") or "").strip()
            stop = (row.get("stop_id") or "").strip()
            route_examples.add(route)
            stop_examples.add(stop)

            service_date = (row.get("service_date") or "").strip()
            monthly_rows[service_date[:7]] += 1

            scheduled_text = (row.get("scheduled") or "").strip()
            actual_text = (row.get("actual") or "").strip()

            scheduled = parse_time(scheduled_text)
            actual = parse_time(actual_text)

            if not scheduled_text:
                scheduled_missing += 1
            elif scheduled is None:
                invalid_timestamps += 1

            if not actual_text:
                actual_missing += 1
            elif actual is None:
                invalid_timestamps += 1

            if scheduled is not None and actual is not None:
                delay = (
                    actual.replace(tzinfo=None) -
                    scheduled.replace(tzinfo=None)
                ).total_seconds() / 60

                delay_counts["valid_delay_rows"] += 1
                if delay < -30:
                    delay_counts["below_minus_30"] += 1
                elif delay > 120:
                    delay_counts["above_120"] += 1
                else:
                    delay_values.append(delay)
                    delay_counts["late_over_3"] += delay > 3
                    delay_counts["on_time_or_early"] += delay <= 3

            key = (
                service_date,
                route,
                (row.get("direction_id") or "").strip(),
                stop,
                scheduled_text,
            )
            key_counts[key] += 1

    print(f"  Rows: {file_rows:,}")

duplicates = sum(count - 1 for count in key_counts.values() if count > 1)

print("\n========== DATA QUALITY SUMMARY ==========")
print(f"Files analyzed: {len(files)}")
print(f"Total rows: {total_rows:,}")
print(f"Duplicate key rows: {duplicates:,}")
print(f"Missing scheduled timestamps: {scheduled_missing:,}")
print(f"Missing actual timestamps: {actual_missing:,}")
print(f"Invalid nonblank timestamps: {invalid_timestamps:,}")

print("\nMissing values by column:")
for column, count in missing.most_common():
    print(f"  {column}: {count:,} ({count / max(total_rows, 1):.1%})")

print("\nDistinct values:")
for column, counts in value_counts.items():
    print(f"\n{column}: {len(counts)} distinct values")
    for value, count in counts.most_common(20):
        print(f"  {value!r}: {count:,}")

print("\nRoute ID examples:", sorted(route_examples)[:30])
print("Stop ID examples:", sorted(stop_examples)[:30])

print("\nRows by service month:")
for month, count in sorted(monthly_rows.items()):
    print(f"  {month}: {count:,}")

print("\nDelay checks (computed from timestamp differences):")
print(f"Rows with both timestamps: {delay_counts['valid_delay_rows']:,}")
print(f"Delays below -30 minutes: {delay_counts['below_minus_30']:,}")
print(f"Delays above 120 minutes: {delay_counts['above_120']:,}")
print(f"Delays greater than 3 minutes among retained delays: "
      f"{delay_counts['late_over_3']:,}")
print(f"Delays 3 minutes or less among retained delays: "
      f"{delay_counts['on_time_or_early']:,}")

if delay_values:
    delay_values.sort()
    n = len(delay_values)
    print("\nRetained delay distribution (after excluding < -30 and > 120):")
    print(f"  Minimum: {delay_values[0]:.2f} minutes")
    print(f"  Median: {delay_values[n // 2]:.2f} minutes")
    print(f"  90th percentile: {delay_values[int((n - 1) * 0.90)]:.2f} minutes")
    print(f"  Maximum: {delay_values[-1]:.2f} minutes")
    print(f"  Late rate: {delay_counts['late_over_3'] / n:.1%}")

print("\nFirst 5 data rows:")
for filename, row in sample_rows:
    print(f"\n{filename}:")
    print(row)

# Save the complete terminal report to a text file.
print("\nAnalysis complete.")
print("Copy the terminal output and share it in the chat.")
