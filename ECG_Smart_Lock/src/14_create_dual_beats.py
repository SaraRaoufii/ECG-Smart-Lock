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

QUALITY_FILE = (
    PROJECT_DIR
    / "results"
    / "beat_quality_flags.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "dual_beats"
)

METADATA_OUTPUT = (
    PROJECT_DIR
    / "results"
    / "dual_beat_metadata.csv"
)


# =========================================================
# 2. تنظیمات ثابت پروژه
# =========================================================

BEFORE_R_SECONDS = 0.2
AFTER_R_SECONDS = 0.4

REFINE_WINDOW_MS = 80


# =========================================================
# 3. فیلتر اصلی
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
# 4. QRS Detection + Refinement
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

        return np.array(
            [],
            dtype=int
        )


    search_samples = int(
        (
            REFINE_WINDOW_MS
            / 1000
        )
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
            abs(
                np.min(segment)
            )
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


        refined_peak = (
            start
            + local_index
        )

        refined.append(
            refined_peak
        )


    return np.array(
        sorted(
            set(
                refined
            )
        ),
        dtype=int
    )


# =========================================================
# 5. استخراج Beatهای منفرد
# =========================================================

def extract_beats(
    signal,
    anchors,
    fs
):

    before_samples = int(
        BEFORE_R_SECONDS
        * fs
    )

    after_samples = int(
        AFTER_R_SECONDS
        * fs
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


        beat = signal[
            start:end
        ]

        beats.append(
            beat
        )


    if len(beats) == 0:

        return np.empty(
            (
                0,
                before_samples
                + after_samples
            )
        )


    return np.array(
        beats
    )


# =========================================================
# 6. خواندن فایل‌ها
# =========================================================

split_df = pd.read_csv(
    SPLIT_FILE
)

quality_df = pd.read_csv(
    QUALITY_FILE
)


# =========================================================
# 7. ساخت پوشه خروجی
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


metadata = []


print(
    "Records to process:",
    len(split_df)
)


# =========================================================
# 8. پردازش Recordها
# =========================================================

for _, row in split_df.iterrows():

    person_id = row[
        "person_id"
    ]

    record_id = row[
        "record_id"
    ]

    split_name = row[
        "split"
    ]


    # -----------------------------------------------------
    # اطلاعات QC همین Record
    # -----------------------------------------------------

    record_quality = quality_df[
        (
            quality_df["person_id"]
            == person_id
        )
        &
        (
            quality_df["record_id"]
            == record_id
        )
    ].copy()


    # -----------------------------------------------------
    # خواندن ECG
    # -----------------------------------------------------

    record_path = (
        DATASET_DIR
        / person_id
        / record_id
    )

    record = wfdb.rdrecord(
        str(
            record_path
        )
    )

    fs = record.fs

    raw_ecg = record.p_signal[
        :,
        0
    ]


    # -----------------------------------------------------
    # فیلتر
    # -----------------------------------------------------

    filtered = biometric_filter(
        raw_ecg,
        fs
    )


    # -----------------------------------------------------
    # Z-score
    # -----------------------------------------------------

    std_value = np.std(
        filtered
    )


    if std_value == 0:

        print(
            "SKIP:",
            person_id,
            record_id,
            "zero STD"
        )

        continue


    biometric_ecg = (
        filtered
        - np.mean(filtered)
    ) / std_value


    # -----------------------------------------------------
    # QRS
    # -----------------------------------------------------

    anchors = detect_and_refine_qrs(
        raw_ecg,
        fs,
        biometric_ecg
    )


    # -----------------------------------------------------
    # Beatها
    # -----------------------------------------------------

    beats = extract_beats(
        biometric_ecg,
        anchors,
        fs
    )


    if len(beats) < 2:

        print(
            "SKIP:",
            person_id,
            record_id,
            "not enough beats"
        )

        continue


    # -----------------------------------------------------
    # وضعیت هر Beat
    # -----------------------------------------------------

    status_map = {}

    for _, qrow in record_quality.iterrows():

        beat_index = int(
            qrow[
                "beat_index"
            ]
        )

        status_map[
            beat_index
        ] = qrow[
            "status"
        ]


    # -----------------------------------------------------
    # پوشه مخصوص Split / Person
    # -----------------------------------------------------

    person_output_dir = (
        OUTPUT_DIR
        / split_name
        / person_id
    )

    person_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # ساخت Dual Beat
    # -----------------------------------------------------

    dual_count = 0


    for i in range(
        len(beats) - 1
    ):

        # index داخل CSV از 1 شروع می‌شود
        beat_1_index = i + 1
        beat_2_index = i + 2


        status_1 = status_map.get(
            beat_1_index,
            "UNKNOWN"
        )

        status_2 = status_map.get(
            beat_2_index,
            "UNKNOWN"
        )


        # هر دو باید GOOD باشند
        if status_1 != "GOOD":
            continue

        if status_2 != "GOOD":
            continue


        beat_1 = beats[
            i
        ]

        beat_2 = beats[
            i + 1
        ]


        # ---------------------------------------------
        # Concatenate
        # ---------------------------------------------

        dual_beat = np.concatenate(
            [
                beat_1,
                beat_2
            ]
        )


        dual_count += 1


        # ---------------------------------------------
        # نام فایل
        # ---------------------------------------------

        filename = (
            f"{person_id}_"
            f"{record_id}_"
            f"dual_{dual_count:03d}.npy"
        )


        output_path = (
            person_output_dir
            / filename
        )


        # ---------------------------------------------
        # ذخیره Dual Beat
        # ---------------------------------------------

        np.save(
            output_path,
            dual_beat
        )


        # ---------------------------------------------
        # Metadata
        # ---------------------------------------------

        metadata.append(
            {
                "person_id": person_id,
                "record_id": record_id,
                "split": split_name,

                "dual_index": dual_count,

                "beat_1_index": (
                    beat_1_index
                ),

                "beat_2_index": (
                    beat_2_index
                ),

                "n_samples": (
                    len(dual_beat)
                ),

                "file_path": str(
                    output_path.relative_to(
                        PROJECT_DIR
                    )
                ),
            }
        )


    print(
        f"{person_id} {record_id}"
        f" | {split_name}"
        f" | Beats={len(beats)}"
        f" | Dual={dual_count}"
    )


# =========================================================
# 9. ذخیره Metadata
# =========================================================

metadata_df = pd.DataFrame(
    metadata
)

metadata_df.to_csv(
    METADATA_OUTPUT,
    index=False
)


# =========================================================
# 10. Summary
# =========================================================

print("\n")
print(
    "===================================="
)

print(
    "DUAL BEAT SUMMARY"
)

print(
    "===================================="
)


print(
    "Total Dual Beats:",
    len(metadata_df)
)


if len(metadata_df) > 0:

    print(
        "\nBy split:"
    )

    print(
        metadata_df[
            "split"
        ].value_counts()
    )


    print(
        "\nDual Beat length:"
    )

    print(
        metadata_df[
            "n_samples"
        ].value_counts()
    )


print(
    "\nSaved metadata:"
)

print(
    METADATA_OUTPUT
)