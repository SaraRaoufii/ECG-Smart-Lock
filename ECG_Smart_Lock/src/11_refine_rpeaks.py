from pathlib import Path

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from scipy.signal import butter, sosfiltfilt


# =========================================================
# 1. مسیرهای پروژه
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "ecgiddb"
    / "ecg-id-database-1.0.0"
)


# =========================================================
# 2. فیلتر اصلی بیومتریک
#
# این همان سیگنالی است که بعداً Beat را از آن استخراج می‌کنیم.
# Band-pass = 0.5 تا 40 Hz
# Butterworth order = 3
# =========================================================

def biometric_filter(
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


# =========================================================
# 3. تابع تشخیص و اصلاح R-Peak
# =========================================================

def detect_and_refine_rpeaks(
    raw_ecg,
    fs,
    biometric_ecg
):

    # -----------------------------------------------------
    # مرحله 1:
    # ساخت سیگنال مخصوص R-Peak Detection
    #
    # این سیگنال فقط برای پیدا کردن محل تقریبی QRS است.
    # Beat نهایی از این سیگنال استخراج نمی‌شود.
    # -----------------------------------------------------

    detection_signal = nk.ecg_clean(
        raw_ecg,
        sampling_rate=fs,
        method="neurokit"
    )


    # -----------------------------------------------------
    # مرحله 2:
    # پیدا کردن Candidate R-Peakها
    # -----------------------------------------------------

    _, info = nk.ecg_peaks(
        detection_signal,
        sampling_rate=fs,
        method="neurokit",
        correct_artifacts=False
    )

    candidate_peaks = np.array(
        info["ECG_R_Peaks"],
        dtype=int
    )


    # اگر اصلاً R پیدا نشد
    if len(candidate_peaks) == 0:

        return (
            candidate_peaks,
            candidate_peaks,
            "unknown"
        )


    # -----------------------------------------------------
    # مرحله 3:
    # محدوده جستجو اطراف Candidate
    #
    # 80 ms قبل و 80 ms بعد
    # -----------------------------------------------------

    search_ms = 80

    search_samples = int(
        (search_ms / 1000)
        * fs
    )


    # -----------------------------------------------------
    # مرحله 4:
    # تشخیص قطبیت کلی ECG
    #
    # می‌خواهیم بفهمیم QRSهای این Record
    # عمدتاً مثبت هستند یا منفی.
    # -----------------------------------------------------

    positive_strengths = []
    negative_strengths = []


    for peak in candidate_peaks:

        start = max(
            0,
            peak - search_samples
        )

        end = min(
            len(biometric_ecg),
            peak + search_samples + 1
        )

        local_segment = biometric_ecg[
            start:end
        ]


        # بیشترین قله مثبت
        positive_strength = np.max(
            local_segment
        )


        # قوی‌ترین قله منفی
        negative_strength = abs(
            np.min(local_segment)
        )


        positive_strengths.append(
            positive_strength
        )

        negative_strengths.append(
            negative_strength
        )


    # -----------------------------------------------------
    # مرحله 5:
    # Median می‌گیریم، نه Mean
    #
    # چون Median نسبت به Artifactهای شدید مقاوم‌تر است.
    # -----------------------------------------------------

    median_positive = np.median(
        positive_strengths
    )

    median_negative = np.median(
        negative_strengths
    )


    if median_positive >= median_negative:

        polarity = "positive"

    else:

        polarity = "negative"


    print(
        "Detected ECG polarity:",
        polarity
    )

    print(
        "Median positive strength:",
        median_positive
    )

    print(
        "Median negative strength:",
        median_negative
    )


    # -----------------------------------------------------
    # مرحله 6:
    # Refinement واقعی R-Peakها
    # -----------------------------------------------------

    refined_peaks = []


    for peak in candidate_peaks:

        start = max(
            0,
            peak - search_samples
        )

        end = min(
            len(biometric_ecg),
            peak + search_samples + 1
        )

        local_segment = biometric_ecg[
            start:end
        ]


        # اگر قطبیت ECG مثبت باشد:
        # قله مثبت را پیدا می‌کنیم.
        if polarity == "positive":

            local_index = np.argmax(
                local_segment
            )


        # اگر ECG واقعاً وارونه باشد:
        # قله منفی را پیدا می‌کنیم.
        else:

            local_index = np.argmin(
                local_segment
            )


        refined_peak = (
            start
            + local_index
        )

        refined_peaks.append(
            refined_peak
        )


    # -----------------------------------------------------
    # مرحله 7:
    # مرتب‌سازی و حذف Peak تکراری احتمالی
    # -----------------------------------------------------

    refined_peaks = np.array(
        sorted(
            set(
                refined_peaks
            )
        ),
        dtype=int
    )


    return (
        candidate_peaks,
        refined_peaks,
        polarity
    )


# =========================================================
# 4. Recordهایی که فعلاً برای کنترل تست می‌کنیم
# =========================================================

records_to_check = [

    ("Person_01", "rec_1"),

    ("Person_14", "rec_1"),
    ("Person_14", "rec_2"),
    ("Person_14", "rec_3"),

]


# =========================================================
# 5. اجرای تست روی Recordها
# =========================================================

for person_id, record_id in records_to_check:

    print("\n")
    print("======================================")
    print(person_id, record_id)
    print("======================================")


    # -----------------------------------------------------
    # خواندن ECG
    # -----------------------------------------------------

    record_path = (
        DATASET_DIR
        / person_id
        / record_id
    )

    record = wfdb.rdrecord(
        str(record_path)
    )

    fs = record.fs

    raw_ecg = record.p_signal[:, 0]


    # -----------------------------------------------------
    # فیلتر اصلی بیومتریک
    # -----------------------------------------------------

    filtered_ecg = biometric_filter(
        raw_ecg,
        fs
    )


    # -----------------------------------------------------
    # Z-score Normalization
    # -----------------------------------------------------

    std_value = np.std(
        filtered_ecg
    )


    if std_value == 0:

        print(
            "ERROR: Standard deviation is zero."
        )

        continue


    biometric_ecg = (
        filtered_ecg
        - np.mean(filtered_ecg)
    ) / std_value


    # -----------------------------------------------------
    # R-Peak Detection + Refinement
    # -----------------------------------------------------

    (
        candidate_peaks,
        refined_peaks,
        polarity

    ) = detect_and_refine_rpeaks(

        raw_ecg,
        fs,
        biometric_ecg
    )


    # -----------------------------------------------------
    # BPM تقریبی
    # -----------------------------------------------------

    duration = (
        len(raw_ecg)
        / fs
    )


    bpm = (
        len(refined_peaks)
        / duration
    ) * 60


    # -----------------------------------------------------
    # نمایش اطلاعات
    # -----------------------------------------------------

    print(
        "Candidate R-peaks:",
        len(candidate_peaks)
    )

    print(
        "Refined R-peaks:",
        len(refined_peaks)
    )

    print(
        "Estimated BPM:",
        bpm
    )

    print(
        "Final polarity:",
        polarity
    )


    # -----------------------------------------------------
    # میزان جابه‌جایی Rها
    # -----------------------------------------------------

    if (
        len(candidate_peaks)
        == len(refined_peaks)
        and len(candidate_peaks) > 0
    ):

        shifts_samples = (
            refined_peaks
            - candidate_peaks
        )

        shifts_ms = (
            shifts_samples
            / fs
        ) * 1000


        print(
            "Median refinement shift (ms):",
            np.median(
                np.abs(
                    shifts_ms
                )
            )
        )


    # -----------------------------------------------------
    # محور زمان
    # -----------------------------------------------------

    time = (
        np.arange(
            len(biometric_ecg)
        )
        / fs
    )


    # -----------------------------------------------------
    # رسم ECG
    # -----------------------------------------------------

    plt.figure(
        figsize=(16, 5)
    )


    plt.plot(
        time,
        biometric_ecg,
        label="Biometric ECG"
    )


    # -----------------------------------------------------
    # Candidate Rها
    # -----------------------------------------------------

    if len(candidate_peaks) > 0:

        plt.scatter(
            candidate_peaks / fs,
            biometric_ecg[
                candidate_peaks
            ],
            marker="x",
            s=60,
            label="Candidate R"
        )


    # -----------------------------------------------------
    # Refined Rها
    # -----------------------------------------------------

    if len(refined_peaks) > 0:

        plt.scatter(
            refined_peaks / fs,
            biometric_ecg[
                refined_peaks
            ],
            marker="o",
            s=55,
            label="Refined R"
        )


    # -----------------------------------------------------
    # عنوان نمودار
    # -----------------------------------------------------

    plt.title(
        f"{person_id} - {record_id} - Refined R-Peaks"
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