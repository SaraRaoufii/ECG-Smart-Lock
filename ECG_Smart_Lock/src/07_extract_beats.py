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


print(
    "Detected R-peaks:",
    len(r_peaks)
)



before_r_seconds = 0.2
after_r_seconds = 0.4

before_r_samples = int(
    before_r_seconds * fs
)

after_r_samples = int(
    after_r_seconds * fs
)

print(
    "Samples before R:",
    before_r_samples
)

print(
    "Samples after R:",
    after_r_samples
)



beats = []

valid_r_peaks = []


for r_peak in r_peaks:

    start = (
        r_peak
        - before_r_samples
    )

    end = (
        r_peak
        + after_r_samples
    )

    if start < 0:
        continue

    if end > len(normalized_ecg):
        continue

    beat = normalized_ecg[
        start:end
    ]

    beats.append(beat)

    valid_r_peaks.append(
        r_peak
    )


beats = np.array(beats)



print(
    "\nValid beats:",
    len(beats)
)

print(
    "Shape of beats:",
    beats.shape
)



number_to_show = min(
    6,
    len(beats)
)


for i in range(number_to_show):

    beat = beats[i]

    time = (
        np.arange(len(beat))
        - before_r_samples
    ) / fs

    plt.figure(
        figsize=(8, 3)
    )

    plt.plot(
        time,
        beat
    )

    plt.axvline(
        x=0,
        linestyle="--",
        label="R-Peak"
    )

    plt.title(
        f"Person 01 - Beat {i + 1}"
    )

    plt.xlabel(
        "Time relative to R (seconds)"
    )

    plt.ylabel(
        "Normalized Amplitude"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()