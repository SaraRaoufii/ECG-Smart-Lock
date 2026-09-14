from pathlib import Path

import wfdb
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "ecgiddb"
    / "ecg-id-database-1.0.0"
)

record_path = (
    DATASET_DIR
    / "Person_01"
    / "rec_1"
)



record = wfdb.rdrecord(str(record_path))

fs = record.fs

raw_ecg = record.p_signal[:, 0]

official_filtered = record.p_signal[:, 1]


print("Sampling rate:", fs)
print("Samples:", len(raw_ecg))


def bandpass_filter(
    signal,
    fs,
    lowcut=0.5,
    highcut=40.0,
    order=3
):

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered_signal = sosfiltfilt(
        sos,
        signal
    )

    return filtered_signal



our_filtered = bandpass_filter(
    raw_ecg,
    fs
)



time = np.arange(len(raw_ecg)) / fs



plt.figure(figsize=(15, 4))

plt.plot(
    time,
    raw_ecg
)

plt.title(
    "Person 01 - Record 1 - RAW ECG"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.show()



plt.figure(figsize=(15, 4))

plt.plot(
    time,
    our_filtered
)

plt.title(
    "Person 01 - Record 1 - Our Bandpass 0.5-40 Hz"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.show()



plt.figure(figsize=(15, 4))

plt.plot(
    time,
    official_filtered,
    label="ECG-ID filtered"
)

plt.plot(
    time,
    our_filtered,
    label="Our 0.5-40 Hz",
    alpha=0.7
)

plt.title(
    "Official Filter vs Our Filter"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()

plt.tight_layout()

plt.show()