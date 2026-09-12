import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks


# =========================================================
# FILE
# =========================================================

INPUT_FILE = Path(
    r"G:\last project\ECG_Smart_Lock\ECG_Smart_Lock\src"
    r"\MAX30003_ECG_128SPS.ino\data\max30003\100"
    r"\rec_11_exercise.csv"
)

FS = 128


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(INPUT_FILE)

ecg = df["ecg"].astype(float).values

print("Samples:", len(ecg))
print("Duration:", len(ecg) / FS, "seconds")


# =========================================================
# FILTER
# =========================================================

def bandpass_filter(signal, fs):

    low = 0.5
    high = 40.0

    nyquist = fs / 2

    b, a = butter(
        3,
        [low / nyquist, high / nyquist],
        btype="band"
    )

    return filtfilt(b, a, signal)


filtered = bandpass_filter(ecg, FS)


# =========================================================
# NORMALIZATION
# =========================================================

filtered = (
    filtered - np.mean(filtered)
) / np.std(filtered)


# =========================================================
# R-PEAK DETECTION
# =========================================================

peaks, properties = find_peaks(
    filtered,
    distance=int(0.4 * FS),
    prominence=0.5
)

print("R-peaks:", len(peaks))


# =========================================================
# SHOW FIRST 10 SECONDS
# =========================================================

seconds = 10

n = seconds * FS

time = np.arange(n) / FS

plt.figure(figsize=(15, 5))

plt.plot(
    time,
    filtered[:n],
    linewidth=1
)

first_peaks = peaks[peaks < n]

plt.scatter(
    first_peaks / FS,
    filtered[first_peaks],
    s=40
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    "Filtered ECG + R Peaks - First 10 Seconds"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# EXTRACT BEATS
# =========================================================

# 250 ms قبل از R
# 450 ms بعد از R

PRE = int(0.25 * FS)
POST = int(0.45 * FS)

beats = []

valid_peaks = []

for peak in peaks:

    start = peak - PRE
    end = peak + POST

    if start < 0:
        continue

    if end >= len(filtered):
        continue

    beat = filtered[start:end]

    beats.append(beat)
    valid_peaks.append(peak)


beats = np.array(beats)

print("Valid beats:", len(beats))
print("Samples per beat:", beats.shape[1])


# =========================================================
# SHOW FIRST 10 BEATS
# =========================================================

plt.figure(figsize=(12, 6))

for i in range(min(10, len(beats))):

    beat_time = np.arange(
        len(beats[i])
    ) / FS

    plt.plot(
        beat_time,
        beats[i],
        alpha=0.7
    )

plt.axvline(
    PRE / FS,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title(
    "First 10 Extracted ECG Beats"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# DUAL BEAT
# =========================================================

dual_beats = []

for i in range(len(beats) - 1):

    dual = np.concatenate(
        [
            beats[i],
            beats[i + 1]
        ]
    )

    dual_beats.append(dual)


dual_beats = np.array(dual_beats)

print(
    "Dual-beats:",
    len(dual_beats)
)

print(
    "Samples per dual-beat:",
    dual_beats.shape[1]
)


# =========================================================
# SHOW FIRST DUAL BEAT
# =========================================================

plt.figure(figsize=(14, 5))

plt.plot(
    dual_beats[0],
    linewidth=1.2
)

plt.axvline(
    len(beats[0]),
    linestyle="--"
)

plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.title(
    "First Dual-Beat"
)

plt.grid(True)

plt.tight_layout()

plt.show()