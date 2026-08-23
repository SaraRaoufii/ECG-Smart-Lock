from pathlib import Path

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from scipy.signal import butter, sosfiltfilt


# =========================================================
# 1. مسیر دیتاست
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "ecgiddb"
    / "ecg-id-database-1.0.0"
)


# =========================================================
# 2. فیلتر
# =========================================================

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


# =========================================================
# 3. Recordهای Person_14
# =========================================================

records_to_check = [
    "rec_1",
    "rec_2",
    "rec_3"
]


for record_id in records_to_check:

    print("\n==============================")
    print("Person_14", record_id)
    print("==============================")


    # -----------------------------------------------------
    # خواندن ECG
    # -----------------------------------------------------

    record_path = (
        DATASET_DIR
        / "Person_14"
        / record_id
    )

    record = wfdb.rdrecord(
        str(record_path)
    )

    fs = record.fs

    raw_ecg = record.p_signal[:, 0]


    # -----------------------------------------------------
    # Filter
    # -----------------------------------------------------

    filtered_ecg = bandpass_filter(
        raw_ecg,
        fs
    )


    # -----------------------------------------------------
    # Z-score
    # -----------------------------------------------------

    normalized_ecg = (
        filtered_ecg
        - np.mean(filtered_ecg)
    ) / np.std(filtered_ecg)


    # -----------------------------------------------------
    # Detection روی ECG معمولی
    # -----------------------------------------------------

    signals_normal, info_normal = nk.ecg_peaks(
        normalized_ecg,
        sampling_rate=fs,
        method="neurokit",
        correct_artifacts=False
    )

    peaks_normal = info_normal[
        "ECG_R_Peaks"
    ]


    # -----------------------------------------------------
    # Detection روی ECG برعکس شده
    #
    # فعلاً فقط برای تشخیص علت مشکل است.
    # -----------------------------------------------------

    signals_inverted, info_inverted = nk.ecg_peaks(
        -normalized_ecg,
        sampling_rate=fs,
        method="neurokit",
        correct_artifacts=False
    )

    peaks_inverted = info_inverted[
        "ECG_R_Peaks"
    ]


    # -----------------------------------------------------
    # چاپ نتیجه
    # -----------------------------------------------------

    print(
        "Normal signal R-peaks:",
        len(peaks_normal)
    )

    print(
        "Inverted signal R-peaks:",
        len(peaks_inverted)
    )

    print(
        "Signal minimum:",
        np.min(normalized_ecg)
    )

    print(
        "Signal maximum:",
        np.max(normalized_ecg)
    )


    # -----------------------------------------------------
    # محور زمان
    # -----------------------------------------------------

    time = (
        np.arange(
            len(normalized_ecg)
        )
        / fs
    )


    # -----------------------------------------------------
    # نمودار ECG اصلی
    # -----------------------------------------------------

    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        time,
        normalized_ecg,
        label="Normalized ECG"
    )


    # Rهایی که روی حالت معمولی پیدا شدند
    if len(peaks_normal) > 0:

        plt.scatter(
            peaks_normal / fs,
            normalized_ecg[
                peaks_normal
            ],
            label="Normal detection"
        )


    plt.title(
        f"Person 14 - {record_id} - Normal Detection"
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


    # -----------------------------------------------------
    # همان ECG اصلی
    # ولی نقاطی که با detector معکوس پیدا شدند
    # -----------------------------------------------------

    plt.figure(
        figsize=(16, 5)
    )

    plt.plot(
        time,
        normalized_ecg,
        label="Original ECG"
    )


    if len(peaks_inverted) > 0:

        plt.scatter(
            peaks_inverted / fs,
            normalized_ecg[
                peaks_inverted
            ],
            label="Peaks found after inversion"
        )


    plt.title(
        f"Person 14 - {record_id} - Inversion Test"
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