"""
preprocessing.py

توابع مشترک پیش‌پردازش سیگنال ECG.
این فایل باید در پوشه‌ی src کنار dataset.py, model.py, train.py قرار بگیرد.

همه‌ی اسکریپت‌های دیگر (چه برای ECG-ID، چه برای داده‌ی جدید آردوینو)
باید از همین‌جا import کنند تا مطمئن شویم دقیقاً یک منطق پردازشی
در کل پروژه استفاده می‌شود.
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt


# =========================================================
# تنظیمات ثابت پروژه
# =========================================================

BEFORE_R_SECONDS = 0.2
AFTER_R_SECONDS = 0.4
REFINE_WINDOW_MS = 80


# =========================================================
# فیلتر اصلی بیومتریک
# =========================================================

def biometric_filter(signal, fs, lowcut=0.5, highcut=40.0, order=3):
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


# =========================================================
# تشخیص QRS و Refinement (با تشخیص خودکار قطبیت)
# =========================================================

def detect_and_refine_qrs(raw_ecg, fs, biometric_ecg):

    detection_signal = nk.ecg_clean(raw_ecg, sampling_rate=fs, method="neurokit")

    _, info = nk.ecg_peaks(
        detection_signal,
        sampling_rate=fs,
        method="neurokit",
        correct_artifacts=False,
    )

    candidates = np.array(info["ECG_R_Peaks"], dtype=int)

    if len(candidates) == 0:
        return np.array([], dtype=int)

    search_samples = int((REFINE_WINDOW_MS / 1000) * fs)

    positive_strengths = []
    negative_strengths = []

    for peak in candidates:
        start = max(0, peak - search_samples)
        end = min(len(biometric_ecg), peak + search_samples + 1)
        segment = biometric_ecg[start:end]
        positive_strengths.append(np.max(segment))
        negative_strengths.append(abs(np.min(segment)))

    median_positive = np.median(positive_strengths)
    median_negative = np.median(negative_strengths)

    polarity = "positive" if median_positive >= median_negative else "negative"

    refined = []

    for peak in candidates:
        start = max(0, peak - search_samples)
        end = min(len(biometric_ecg), peak + search_samples + 1)
        segment = biometric_ecg[start:end]

        local_index = np.argmax(segment) if polarity == "positive" else np.argmin(segment)
        refined.append(start + local_index)

    return np.array(sorted(set(refined)), dtype=int)


# =========================================================
# استخراج Beatهای منفرد
# =========================================================

def extract_beats(signal, anchors, fs):

    before_samples = int(BEFORE_R_SECONDS * fs)
    after_samples = int(AFTER_R_SECONDS * fs)

    beats = []

    for anchor in anchors:
        start = anchor - before_samples
        end = anchor + after_samples

        if start < 0 or end > len(signal):
            continue

        beats.append(signal[start:end])

    if len(beats) == 0:
        return np.empty((0, before_samples + after_samples))

    return np.array(beats)