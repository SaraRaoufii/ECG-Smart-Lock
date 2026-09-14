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

    return sosfiltfilt(
        sos,
        signal
    )


filtered_ecg = bandpass_filter(
    raw_ecg,
    fs
)


mean_value = np.mean(filtered_ecg)

std_value = np.std(filtered_ecg)

normalized_ecg = (
    filtered_ecg - mean_value
) / std_value



print("Before normalization:")
print("Mean:", np.mean(filtered_ecg))
print("STD :", np.std(filtered_ecg))

print("\nAfter normalization:")
print("Mean:", np.mean(normalized_ecg))
print("STD :", np.std(normalized_ecg))


time = np.arange(
    len(normalized_ecg)
) / fs



plt.figure(figsize=(15, 4))

plt.plot(
    time,
    filtered_ecg
)

plt.title(
    "Filtered ECG - Before Z-score"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()



plt.figure(figsize=(15, 4))

plt.plot(
    time,
    normalized_ecg
)

plt.title(
    "Filtered ECG - After Z-score"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")

plt.tight_layout()
plt.show()