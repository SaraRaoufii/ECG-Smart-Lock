from pathlib import Path

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk

from scipy.signal import butter, sosfiltfilt


# =========================================================
# 1. مسیرها
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "ecgiddb"
    / "ecg-id-database-1.0.0"
)

BEAT_FLAGS_FILE = (
    PROJECT_DIR
    / "results"
    / "beat_quality_flags.csv"
)


# =========================================================
# 2. تنظیمات ثابت پروژه
# =========================================================

BEFORE_R_SECONDS = 0.2
AFTER_R_SECONDS = 0.4
REFINE_WINDOW_MS = 80


# =========================================================
# 3. فیلتر اصلی بیومتریک
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

    return sosfiltfilt(
        sos,
        signal
    )


# =========================================================
# 4. تشخیص QRS و Refinement
# =========================================================

def detect_and_refine_qrs(
    raw_ecg,
    fs,
    biometric_ecg
):

    detection_signal = nk.ecg_clean(
        raw_ecg,
        sampling_rate=fs,
        method="neurokit"
    )

    _, info = nk.ecg_peaks(
        detection_signal,
        sampling_rate=fs,
        method="neurokit",
        correct_artifacts=False
    )

    candidates = np.array(
        info["ECG_R_Peaks"],
        dtype=int
    )

    if len(candidates) == 0:

        return np.array([], dtype=int)


    search_samples = int(
        (REFINE_WINDOW_MS / 1000)
        * fs
    )


    positive_strengths = []
    negative_strengths = []


    for peak in candidates:

        start = max(
            0,
            peak - search_samples
        )

        end = min(
            len(biometric_ecg),
            peak + search_samples + 1
        )

        segment = biometric_ecg[
            start:end
        ]

        positive_strengths.append(
            np.max(segment)
        )

        negative_strengths.append(
            abs(np.min(segment))
        )


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


    refined = []


    for peak in candidates:

        start = max(
            0,
            peak - search_samples
        )

        end = min(
            len(biometric_ecg),
            peak + search_samples + 1
        )

        segment = biometric_ecg[
            start:end
        ]


        if polarity == "positive":

            local_index = np.argmax(
                segment
            )

        else:

            local_index = np.argmin(
                segment
            )


        refined.append(
            start + local_index
        )


    return np.array(
        sorted(set(refined)),
        dtype=int
    )


# =========================================================
# 5. استخراج Beatها
# =========================================================

def extract_beats(
    signal,
    anchors,
    fs
):

    before_samples = int(
        BEFORE_R_SECONDS * fs
    )

    after_samples = int(
        AFTER_R_SECONDS * fs
    )

    beats = []


    for anchor in anchors:

        start = (
            anchor
            - before_samples
        )

        end = (
            anchor
            + after_samples
        )


        if start < 0:
            continue

        if end > len(signal):
            continue


        beats.append(
            signal[start:end]
        )


    return np.array(beats)


# =========================================================
# 6. فایل Flagها
# =========================================================

flags_df = pd.read_csv(
    BEAT_FLAGS_FILE
)


# فقط دو Record مشکوک
records_to_check = [

    ("Person_85", "rec_1"),

    ("Person_88", "rec_1"),

]


# =========================================================
# 7. بررسی
# =========================================================

for person_id, record_id in records_to_check:

    print("\n")
    print(
        "=================================="
    )

    print(
        person_id,
        record_id
    )

    print(
        "=================================="
    )


    # -----------------------------------------------------
    # Flagهای همین Record
    # -----------------------------------------------------

    record_flags = flags_df[
        (
            flags_df["person_id"]
            == person_id
        )
        &
        (
            flags_df["record_id"]
            == record_id
        )
    ].copy()


    flagged_rows = record_flags[
        record_flags["status"]
        != "GOOD"
    ]


    print(
        "Total beats in CSV:",
        len(record_flags)
    )

    print(
        "Flagged beats:",
        len(flagged_rows)
    )


    print("\nFlag details:")

    print(
        flagged_rows[
            [
                "beat_index",
                "correlation",
                "max_abs_amplitude",
                "status",
            ]
        ].to_string(
            index=False
        )
    )


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
    # Preprocessing
    # -----------------------------------------------------

    filtered = biometric_filter(
        raw_ecg,
        fs
    )

    biometric_ecg = (
        filtered
        - np.mean(filtered)
    ) / np.std(filtered)


    # -----------------------------------------------------
    # QRS
    # -----------------------------------------------------

    anchors = detect_and_refine_qrs(
        raw_ecg,
        fs,
        biometric_ecg
    )


    beats = extract_beats(
        biometric_ecg,
        anchors,
        fs
    )


    # -----------------------------------------------------
    # محور زمان Beat
    # -----------------------------------------------------

    before_samples = int(
        BEFORE_R_SECONDS * fs
    )

    beat_time = (
        np.arange(
            beats.shape[1]
        )
        - before_samples
    ) / fs


    # -----------------------------------------------------
    # نمایش Beatهای Flag شده
    # -----------------------------------------------------

    for _, row in flagged_rows.iterrows():

        beat_number = int(
            row["beat_index"]
        )

        # CSV از 1 شروع شده
        array_index = (
            beat_number - 1
        )


        if (
            array_index < 0
            or
            array_index >= len(beats)
        ):

            continue


        beat = beats[
            array_index
        ]


        plt.figure(
            figsize=(9, 4)
        )

        plt.plot(
            beat_time,
            beat
        )

        plt.axvline(
            0,
            linestyle="--",
            label="QRS Anchor"
        )


        plt.title(
            f"{person_id} - {record_id}"
            f" - FLAGGED Beat {beat_number}"
            f"\n{row['status']}"
            f" | Corr={row['correlation']:.3f}"
            f" | Amp={row['max_abs_amplitude']:.3f}"
        )


        plt.xlabel(
            "Time relative to QRS anchor (s)"
        )

        plt.ylabel(
            "Normalized Amplitude"
        )

        plt.legend()

        plt.tight_layout()

        plt.show()