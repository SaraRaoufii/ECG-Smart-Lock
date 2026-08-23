from pathlib import Path

import wfdb
import numpy as np
import pandas as pd
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

SPLIT_FILE = (
    PROJECT_DIR
    / "results"
    / "dataset_split.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "record_quality_check.csv"
)


# =========================================================
# 2. Band-pass Filter
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
# 3. خواندن Split
# =========================================================

df = pd.read_csv(SPLIT_FILE)

print("Records to check:", len(df))


# =========================================================
# 4. لیست نتایج
# =========================================================

results = []


# =========================================================
# 5. پردازش تک‌تک Recordها
# =========================================================

for row_number, row in df.iterrows():

    person_id = row["person_id"]
    record_id = row["record_id"]
    split = row["split"]

    record_path = (
        DATASET_DIR
        / person_id
        / record_id
    )

    try:

        # -----------------------------
        # خواندن ECG
        # -----------------------------

        record = wfdb.rdrecord(
            str(record_path)
        )

        fs = record.fs

        raw_ecg = record.p_signal[:, 0]


        # -----------------------------
        # Filter
        # -----------------------------

        filtered_ecg = bandpass_filter(
            raw_ecg,
            fs
        )


        # -----------------------------
        # Z-score
        # -----------------------------

        std_value = np.std(filtered_ecg)

        if std_value == 0:

            raise ValueError(
                "Signal standard deviation is zero."
            )

        normalized_ecg = (
            filtered_ecg
            - np.mean(filtered_ecg)
        ) / std_value


        # -----------------------------
        # R-Peak Detection
        # -----------------------------

        signals, info = nk.ecg_peaks(
            normalized_ecg,
            sampling_rate=fs,
            method="neurokit",
            correct_artifacts=True
        )

        r_peaks = info[
            "ECG_R_Peaks"
        ]


        # -----------------------------
        # Beat Window
        # -----------------------------

        before_r_samples = int(
            0.2 * fs
        )

        after_r_samples = int(
            0.4 * fs
        )


        # -----------------------------
        # شمارش Beatهای کامل
        # -----------------------------

        valid_beats = 0

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

            if end > len(
                normalized_ecg
            ):
                continue

            valid_beats += 1


        # -----------------------------
        # BPM تقریبی
        # -----------------------------

        duration_seconds = (
            len(normalized_ecg)
            / fs
        )

        estimated_bpm = (
            len(r_peaks)
            / duration_seconds
        ) * 60


        # -----------------------------
        # ذخیره نتیجه
        # -----------------------------

        results.append(
            {
                "person_id": person_id,
                "record_id": record_id,
                "split": split,
                "r_peaks": len(r_peaks),
                "valid_beats": valid_beats,
                "estimated_bpm": estimated_bpm,
                "status": "OK",
            }
        )


        print(
            f"{person_id} {record_id}"
            f" | {split}"
            f" | R={len(r_peaks)}"
            f" | Beats={valid_beats}"
            f" | BPM={estimated_bpm:.1f}"
        )


    except Exception as e:

        results.append(
            {
                "person_id": person_id,
                "record_id": record_id,
                "split": split,
                "r_peaks": np.nan,
                "valid_beats": np.nan,
                "estimated_bpm": np.nan,
                "status": f"ERROR: {e}",
            }
        )

        print(
            "ERROR:",
            person_id,
            record_id,
            e
        )


# =========================================================
# 6. تبدیل نتایج به DataFrame
# =========================================================

quality_df = pd.DataFrame(
    results
)


# =========================================================
# 7. ذخیره CSV
# =========================================================

quality_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 8. خلاصه
# =========================================================

print("\n==============================")
print("SUMMARY")
print("==============================")

print(
    "Total records:",
    len(quality_df)
)

print(
    "Successful:",
    (
        quality_df["status"]
        == "OK"
    ).sum()
)

print(
    "Errors:",
    (
        quality_df["status"]
        != "OK"
    ).sum()
)


print("\nValid beats statistics:")

print(
    quality_df[
        quality_df["status"] == "OK"
    ]["valid_beats"].describe()
)


print("\nBPM statistics:")

print(
    quality_df[
        quality_df["status"] == "OK"
    ]["estimated_bpm"].describe()
)


print("\nSaved:")
print(OUTPUT_FILE)