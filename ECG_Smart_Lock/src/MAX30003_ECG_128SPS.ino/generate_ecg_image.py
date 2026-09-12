import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, resample


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = Path(
    r"G:\last project\ECG_Smart_Lock\ECG_Smart_Lock\src"
    r"\MAX30003_ECG_128SPS.ino\data\max30003\100"
    r"\rec_11_exercise.csv"
)

OUTPUT_DIR = INPUT_FILE.parent / "images"

FS = 128
TARGET_FS = 500

IMAGE_SIZE = 112

# تعداد نمونه برای هر تصویر
TARGET_SAMPLES = 600


# =========================================================
# 1. LOAD ECG
# =========================================================

print("=" * 60)
print("ECG IMAGE GENERATION")
print("=" * 60)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

if "ecg" not in df.columns:
    raise ValueError(
        "Column 'ecg' was not found in CSV."
    )

ecg = df["ecg"].astype(float).values

print("Samples:", len(ecg))
print("Sampling rate:", FS, "Hz")


# =========================================================
# 2. REMOVE DC OFFSET
# =========================================================

ecg = ecg - np.mean(ecg)


# =========================================================
# 3. BANDPASS FILTER
# =========================================================

def bandpass_filter(signal, fs):

    lowcut = 0.5
    highcut = 40.0

    nyquist = fs / 2

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        3,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


filtered_ecg = bandpass_filter(
    ecg,
    FS
)


# =========================================================
# 4. NORMALIZE
# =========================================================

std = np.std(filtered_ecg)

if std > 0:
    normalized_ecg = (
        filtered_ecg - np.mean(filtered_ecg)
    ) / std
else:
    normalized_ecg = filtered_ecg


# =========================================================
# 5. R-PEAK DETECTION
# =========================================================

# حداقل فاصله بین دو R peak
# حدود 0.4 ثانیه = حداکثر حدود 150 BPM

min_distance = int(
    0.4 * FS
)

# prominence بر اساس دامنه سیگنال
prominence = 0.5 * np.std(
    normalized_ecg
)

peaks, properties = find_peaks(
    normalized_ecg,
    distance=min_distance,
    prominence=prominence
)

print("\nR-peaks detected:", len(peaks))


# =========================================================
# 6. SHOW ECG + R PEAKS
# =========================================================

time = np.arange(
    len(normalized_ecg)
) / FS

plt.figure(figsize=(14, 5))

plt.plot(
    time,
    normalized_ecg,
    linewidth=0.8
)

plt.scatter(
    peaks / FS,
    normalized_ecg[peaks],
    s=20
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title(
    "Person 100 - rec_11_exercise - R Peak Detection"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# =========================================================
# 7. EXTRACT BEATS
# =========================================================

# محدوده اطراف R peak
PRE = 0.25
POST = 0.45

pre_samples = int(
    PRE * FS
)

post_samples = int(
    POST * FS
)

beats = []

for peak in peaks:

    start = peak - pre_samples
    end = peak + post_samples

    if start < 0:
        continue

    if end >= len(normalized_ecg):
        continue

    beat = normalized_ecg[start:end]

    beats.append(beat)


beats = np.array(beats)

print(
    "Valid beats extracted:",
    len(beats)
)

if len(beats) < 2:
    raise RuntimeError(
        "Not enough valid beats detected."
    )


# =========================================================
# 8. BUILD DUAL-BEATS
# =========================================================

dual_beats = []

for i in range(len(beats) - 1):

    beat1 = beats[i]
    beat2 = beats[i + 1]

    dual = np.concatenate(
        [beat1, beat2]
    )

    dual_beats.append(dual)


dual_beats = np.array(
    dual_beats
)

print(
    "Dual-beats generated:",
    len(dual_beats)
)


# =========================================================
# 9. RESAMPLE DUAL-BEATS TO 500 Hz
# =========================================================

resampled_dual_beats = []

for dual in dual_beats:

    duration = len(dual) / FS

    target_length = int(
        round(duration * TARGET_FS)
    )

    resampled = resample(
        dual,
        target_length
    )

    resampled_dual_beats.append(
        resampled
    )


# =========================================================
# 10. CREATE 600-SAMPLE SEGMENTS
# =========================================================

segments = []

for signal in resampled_dual_beats:

    if len(signal) < TARGET_SAMPLES:
        continue

    # از ابتدای dual beat
    # یک پنجره 600 نمونه‌ای بردار

    segment = signal[
        :TARGET_SAMPLES
    ]

    segments.append(segment)


segments = np.array(
    segments
)

print(
    "600-sample segments:",
    len(segments)
)

if len(segments) == 0:
    raise RuntimeError(
        "No 600-sample segments were generated."
    )


# =========================================================
# 11. CREATE OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 12. CONVERT 1D SIGNAL TO 2D IMAGE
# =========================================================

def signal_to_image(
    signal,
    size=112
):

    # Normalize to [0, 1]

    minimum = np.min(signal)
    maximum = np.max(signal)

    if maximum - minimum == 0:

        normalized = np.zeros_like(
            signal
        )

    else:

        normalized = (
            signal - minimum
        ) / (
            maximum - minimum
        )

    # تبدیل 600 نمونه به تصویر 112×112
    #
    # برای اینکه شکل موج حفظ شود:
    # ابتدا interpolate می‌کنیم

    x_old = np.linspace(
        0,
        1,
        len(normalized)
    )

    x_new = np.linspace(
        0,
        1,
        size * size
    )

    resized = np.interp(
        x_new,
        x_old,
        normalized
    )

    image = resized.reshape(
        size,
        size
    )

    return image


# =========================================================
# 13. GENERATE IMAGES
# =========================================================

generated_images = []

for i, segment in enumerate(segments):

    image = signal_to_image(
        segment,
        IMAGE_SIZE
    )

    output_file = (
        OUTPUT_DIR
        / f"rec_11_exercise_dual_{i+1:04d}.png"
    )

    plt.imsave(
        output_file,
        image,
        cmap="gray"
    )

    generated_images.append(
        output_file
    )


# =========================================================
# 14. SHOW FIRST GENERATED IMAGE
# =========================================================

first_image = signal_to_image(
    segments[0],
    IMAGE_SIZE
)

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    first_image,
    cmap="gray",
    aspect="equal"
)

plt.axis("off")

plt.title(
    "ECG Image - 112 × 112"
)

plt.tight_layout()

plt.show()


# =========================================================
# 15. FINAL REPORT
# =========================================================

print("\n" + "=" * 60)
print("IMAGE GENERATION COMPLETE")
print("=" * 60)

print("Input:")
print(INPUT_FILE)

print(
    "R-peaks:",
    len(peaks)
)

print(
    "Beats:",
    len(beats)
)

print(
    "Dual-beats:",
    len(dual_beats)
)

print(
    "Images generated:",
    len(generated_images)
)

print(
    "Image size:",
    f"{IMAGE_SIZE} × {IMAGE_SIZE}"
)

print(
    "Output directory:"
)

print(OUTPUT_DIR)

print("=" * 60)