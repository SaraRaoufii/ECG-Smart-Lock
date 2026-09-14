from pathlib import Path

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

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



normalized_ecg = (
    filtered_ecg
    - np.mean(filtered_ecg)
) / np.std(filtered_ecg)



signals, info = nk.ecg_peaks(
    normalized_ecg,
    sampling_rate=fs,
    method="neurokit",
    correct_artifacts=True
)

r_peaks = info["ECG_R_Peaks"]



print("Sampling rate:", fs)

print(
    "Number of detected R-peaks:",
    len(r_peaks)
)

print(
    "R-peak sample positions:"
)

print(r_peaks)


r_peak_times = r_peaks / fs

print(
    "\nR-peak times in seconds:"
)

print(r_peak_times)



duration_seconds = len(
    normalized_ecg
) / fs

estimated_bpm = (
    len(r_peaks)
    / duration_seconds
) * 60

print(
    "\nEstimated heart rate:",
    estimated_bpm,
    "BPM"
)



time = np.arange(
    len(normalized_ecg)
) / fs



plt.figure(
    figsize=(16, 5)
)

plt.plot(
    time,
    normalized_ecg,
    label="Normalized ECG"
)

plt.scatter(
    r_peak_times,
    normalized_ecg[r_peaks],
    marker="o",
    label="Detected R-Peaks"
)

plt.title(
    "Person 01 - Record 1 - R-Peak Detection"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Normalized Amplitude"
)

plt.legend()

plt.tight_layout()

plt.show()