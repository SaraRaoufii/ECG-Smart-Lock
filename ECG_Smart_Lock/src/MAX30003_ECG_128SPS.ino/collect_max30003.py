import serial
import csv
import time
from pathlib import Path
import re


# =========================================================
# Settings
# =========================================================

PORT = "COM5"
BAUDRATE = 115200

SAMPLING_RATE = 128
DURATION_SECONDS = 60

EXPECTED_SAMPLES = SAMPLING_RATE * DURATION_SECONDS

BASE_OUTPUT_DIR = Path("data") / "max30003"


# =========================================================
# Get Person ID
# =========================================================

person_id = input(
    "Person ID (مثلاً Person_01): "
).strip()


if not person_id:
    print("ERROR: Person ID cannot be empty.")
    raise SystemExit


# =========================================================
# Create person folder
# =========================================================

person_dir = BASE_OUTPUT_DIR / person_id

person_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Find next recording number
# =========================================================

record_numbers = []

for file in person_dir.glob("rec_*.csv"):

    match = re.search(
        r"rec_(\d+)",
        file.name
    )

    if match:
        record_numbers.append(
            int(match.group(1))
        )


if record_numbers:
    record_number = max(record_numbers) + 1
else:
    record_number = 1


# =========================================================
# Condition
# =========================================================

condition = input(
    "Condition "
    "(rest / after_walking / after_exercise / recovery): "
).strip()


if not condition:
    condition = "unknown"


# Replace spaces with underscore

condition = condition.replace(
    " ",
    "_"
)


# =========================================================
# Output filename
# =========================================================

output_file = (
    person_dir
    / f"rec_{record_number:02d}_{condition}.csv"
)


# =========================================================
# Show information before starting
# =========================================================

print()
print("=" * 55)
print("MAX30003 ECG DATA COLLECTION")
print("=" * 55)

print("Person       :", person_id)
print("Recording    :", f"{record_number:02d}")
print("Condition    :", condition)
print("Sampling rate:", f"{SAMPLING_RATE} Hz")
print("Duration     :", f"{DURATION_SECONDS} seconds")
print("Expected     :", f"{EXPECTED_SAMPLES} samples")
print("Output       :", output_file)

print("=" * 55)
print()


# =========================================================
# Open Serial Port
# =========================================================

try:

    ser = serial.Serial(
        PORT,
        BAUDRATE,
        timeout=2
    )

except serial.SerialException as e:

    print()
    print("ERROR: Could not open serial port.")
    print(e)

    raise SystemExit


# =========================================================
# Give Arduino time to stabilize
# =========================================================

print("Connected to:", PORT)

time.sleep(2)


# Clear old data already waiting in Serial buffer

ser.reset_input_buffer()


print()
print("Get ready...")
time.sleep(3)

print()
print("STARTING ECG COLLECTION...")
print()


# =========================================================
# Data collection
# =========================================================

samples = []

collection_started = False

start_time = None


while len(samples) < EXPECTED_SAMPLES:

    line = ser.readline().decode(
        "utf-8",
        errors="ignore"
    ).strip()


    # Ignore empty lines

    if not line:
        continue


    # Try to convert Arduino output to integer

    try:

        ecg_value = int(line)

    except ValueError:

        continue


    # Start timing when the FIRST valid ECG sample arrives

    if not collection_started:

        collection_started = True

        start_time = time.perf_counter()

        print("First valid ECG sample received.")
        print("Recording...")


    # -----------------------------------------------------
    # Sample index
    # -----------------------------------------------------

    sample_index = len(samples)


    # -----------------------------------------------------
    # Timestamp based on SENSOR sampling rate
    #
    # NOT PC/Serial arrival time
    # -----------------------------------------------------

    timestamp = sample_index / SAMPLING_RATE


    # -----------------------------------------------------
    # Save sample
    # -----------------------------------------------------

    samples.append(
        (
            sample_index,
            timestamp,
            ecg_value
        )
    )


# =========================================================
# Close Serial
# =========================================================

ser.close()


# =========================================================
# Save CSV
# =========================================================

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "sample_index",
            "timestamp",
            "ecg"
        ]
    )

    writer.writerows(samples)


# =========================================================
# Final information
# =========================================================

print()
print("=" * 55)
print("ECG COLLECTION COMPLETE")
print("=" * 55)

print(
    "Samples collected:",
    len(samples)
)

print(
    "Expected samples :",
    EXPECTED_SAMPLES
)

print(
    "Sampling rate    :",
    f"{SAMPLING_RATE} Hz"
)

print(
    "Duration         :",
    f"{DURATION_SECONDS} seconds"
)

print(
    "Person           :",
    person_id
)

print(
    "Recording        :",
    f"{record_number:02d}"
)

print(
    "Condition        :",
    condition
)

print(
    "Saved to         :",
    output_file
)

print("=" * 55)