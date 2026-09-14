from pathlib import Path

import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk

from scipy.signal import butter, sosfiltfilt



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

RECORD_OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "final_record_quality.csv"
)

BEAT_OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "beat_quality_flags.csv"
)



BEFORE_R_SECONDS = 0.2

AFTER_R_SECONDS = 0.4

REFINE_WINDOW_MS = 80

CORRELATION_THRESHOLD = 0.50


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

    candidate_peaks = np.array(
        info["ECG_R_Peaks"],
        dtype=int
    )


    if len(candidate_peaks) == 0:

        return (
            candidate_peaks,
            candidate_peaks,
            "unknown"
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


        positive_strengths.append(
            np.max(local_segment)
        )

        negative_strengths.append(
            abs(
                np.min(local_segment)
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


        if polarity == "positive":

            local_index = np.argmax(
                local_segment
            )

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



def extract_beats(
    biometric_ecg,
    qrs_anchors,
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

    used_anchors = []


    for anchor in qrs_anchors:

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

        if end > len(
            biometric_ecg
        ):
            continue


        beat = biometric_ecg[
            start:end
        ]


        beats.append(
            beat
        )

        used_anchors.append(
            anchor
        )


    if len(beats) == 0:

        return (
            np.empty(
                (
                    0,
                    before_samples
                    + after_samples
                )
            ),
            np.array(
                [],
                dtype=int
            )
        )


    return (
        np.array(beats),
        np.array(
            used_anchors,
            dtype=int
        )
    )



def safe_correlation(
    x,
    y
):

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan


    correlation = np.corrcoef(
        x,
        y
    )[0, 1]


    return correlation



def evaluate_beats(
    beats
):

    if len(beats) == 0:

        return (
            [],
            np.array([]),
            np.array([]),
            np.array([])
        )



    median_template = np.median(
        beats,
        axis=0
    )



    correlations = np.array(
        [
            safe_correlation(
                beat,
                median_template
            )
            for beat in beats
        ]
    )



    amplitudes = np.array(
        [
            np.max(
                np.abs(
                    beat
                )
            )
            for beat in beats
        ]
    )



    median_amp = np.median(
        amplitudes
    )

    mad_amp = np.median(
        np.abs(
            amplitudes
            - median_amp
        )
    )



    if mad_amp > 0:

        robust_sigma = (
            1.4826
            * mad_amp
        )

        amplitude_threshold = (
            median_amp
            + 5
            * robust_sigma
        )

    else:

        amplitude_threshold = (
            median_amp
            * 3
        )


    flags = []


    for i in range(
        len(beats)
    ):

        reasons = []


        if (
            np.isnan(
                correlations[i]
            )
            or
            correlations[i]
            < CORRELATION_THRESHOLD
        ):

            reasons.append(
                "low_correlation"
            )


        if (
            amplitudes[i]
            > amplitude_threshold
        ):

            reasons.append(
                "amplitude_outlier"
            )



        if len(reasons) == 0:

            status = "GOOD"

        else:

            status = (
                "FLAGGED:"
                + "|".join(
                    reasons
                )
            )


        flags.append(
            status
        )


    return (
        flags,
        correlations,
        amplitudes,
        median_template
    )


split_df = pd.read_csv(
    SPLIT_FILE
)


print(
    "Total records to process:",
    len(split_df)
)



record_results = []

beat_results = []



for row_index, row in split_df.iterrows():

    person_id = row[
        "person_id"
    ]

    record_id = row[
        "record_id"
    ]

    split_name = row[
        "split"
    ]


    record_path = (
        DATASET_DIR
        / person_id
        / record_id
    )


    try:


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



        filtered_ecg = biometric_filter(
            raw_ecg,
            fs
        )


        std_value = np.std(
            filtered_ecg
        )


        if std_value == 0:

            raise ValueError(
                "Zero standard deviation"
            )


        biometric_ecg = (
            filtered_ecg
            - np.mean(
                filtered_ecg
            )
        ) / std_value



        (
            candidate_peaks,
            qrs_anchors,
            polarity

        ) = detect_and_refine_qrs(

            raw_ecg,
            fs,
            biometric_ecg
        )



        (
            beats,
            used_anchors

        ) = extract_beats(

            biometric_ecg,
            qrs_anchors,
            fs
        )


        # -------------------------------------------------
        # Beat Quality
        # -------------------------------------------------

        (
            flags,
            correlations,
            amplitudes,
            median_template

        ) = evaluate_beats(
            beats
        )


        # -------------------------------------------------
        # شمارش Good / Flagged
        # -------------------------------------------------

        good_count = sum(
            flag == "GOOD"
            for flag in flags
        )

        flagged_count = (
            len(flags)
            - good_count
        )


        if len(flags) > 0:

            flagged_ratio = (
                flagged_count
                / len(flags)
            )

        else:

            flagged_ratio = 1.0


        # -------------------------------------------------
        # BPM
        # -------------------------------------------------

        duration_seconds = (
            len(raw_ecg)
            / fs
        )


        estimated_bpm = (

            len(qrs_anchors)
            / duration_seconds

        ) * 60


        # -------------------------------------------------
        # Record Status
        #
        # اینها فقط Flag اولیه‌اند.
        # -------------------------------------------------

        suspicious_reasons = []


        if len(qrs_anchors) < 10:

            suspicious_reasons.append(
                "few_qrs"
            )


        if good_count < 10:

            suspicious_reasons.append(
                "few_good_beats"
            )


        if flagged_ratio > 0.25:

            suspicious_reasons.append(
                "many_flagged_beats"
            )


        if len(
            suspicious_reasons
        ) == 0:

            record_status = "OK"

        else:

            record_status = (
                "CHECK:"
                + "|".join(
                    suspicious_reasons
                )
            )


        # -------------------------------------------------
        # ذخیره Record Level
        # -------------------------------------------------

        record_results.append(
            {
                "person_id": person_id,
                "record_id": record_id,
                "split": split_name,
                "polarity": polarity,
                "candidate_qrs": len(
                    candidate_peaks
                ),
                "refined_qrs": len(
                    qrs_anchors
                ),
                "valid_beats": len(
                    beats
                ),
                "good_beats": good_count,
                "flagged_beats": flagged_count,
                "flagged_ratio": flagged_ratio,
                "estimated_bpm": estimated_bpm,
                "record_status": record_status,
            }
        )


        # -------------------------------------------------
        # ذخیره Beat Level
        # -------------------------------------------------

        for beat_index in range(
            len(beats)
        ):

            beat_results.append(
                {
                    "person_id": person_id,
                    "record_id": record_id,
                    "split": split_name,

                    "beat_index": (
                        beat_index
                        + 1
                    ),

                    "qrs_sample": int(
                        used_anchors[
                            beat_index
                        ]
                    ),

                    "qrs_time_seconds": (
                        used_anchors[
                            beat_index
                        ]
                        / fs
                    ),

                    "correlation": (
                        correlations[
                            beat_index
                        ]
                    ),

                    "max_abs_amplitude": (
                        amplitudes[
                            beat_index
                        ]
                    ),

                    "status": flags[
                        beat_index
                    ],
                }
            )


        # -------------------------------------------------
        # چاپ هر Record
        # -------------------------------------------------

        print(
            f"{person_id} {record_id}"
            f" | {split_name}"
            f" | Pol={polarity}"
            f" | QRS={len(qrs_anchors)}"
            f" | Good={good_count}"
            f" | Flagged={flagged_count}"
            f" | BPM={estimated_bpm:.1f}"
            f" | {record_status}"
        )


    except Exception as e:

        print(
            "ERROR:",
            person_id,
            record_id,
            e
        )


        record_results.append(
            {
                "person_id": person_id,
                "record_id": record_id,
                "split": split_name,
                "polarity": "unknown",
                "candidate_qrs": np.nan,
                "refined_qrs": np.nan,
                "valid_beats": np.nan,
                "good_beats": np.nan,
                "flagged_beats": np.nan,
                "flagged_ratio": np.nan,
                "estimated_bpm": np.nan,
                "record_status": (
                    "ERROR:"
                    + str(e)
                ),
            }
        )


# =========================================================
# 11. DataFrames
# =========================================================

record_df = pd.DataFrame(
    record_results
)

beat_df = pd.DataFrame(
    beat_results
)


# =========================================================
# 12. ذخیره CSVها
# =========================================================

record_df.to_csv(
    RECORD_OUTPUT_FILE,
    index=False
)

beat_df.to_csv(
    BEAT_OUTPUT_FILE,
    index=False
)


# =========================================================
# 13. Summary
# =========================================================

print("\n")
print("====================================")
print("FINAL QUALITY SUMMARY")
print("====================================")


print(
    "Total records:",
    len(record_df)
)


print(
    "OK records:",
    (
        record_df[
            "record_status"
        ]
        == "OK"
    ).sum()
)


print(
    "Records needing check:",
    (
        record_df[
            "record_status"
        ]
        != "OK"
    ).sum()
)


print(
    "\nTotal extracted beats:",
    len(beat_df)
)


if len(
    beat_df
) > 0:

    print(
        "Good beats:",
        (
            beat_df[
                "status"
            ]
            == "GOOD"
        ).sum()
    )


    print(
        "Flagged beats:",
        (
            beat_df[
                "status"
            ]
            != "GOOD"
        ).sum()
    )


# =========================================================
# 14. Recordهای مشکوک
# =========================================================

suspicious_df = record_df[
    record_df[
        "record_status"
    ] != "OK"
]


print(
    "\nRecords needing manual check:"
)


if len(
    suspicious_df
) == 0:

    print(
        "None"
    )

else:

    print(
        suspicious_df[
            [
                "person_id",
                "record_id",
                "split",
                "polarity",
                "refined_qrs",
                "good_beats",
                "flagged_beats",
                "estimated_bpm",
                "record_status",
            ]
        ].to_string(
            index=False
        )
    )


print("\nSaved:")

print(
    RECORD_OUTPUT_FILE
)

print(
    BEAT_OUTPUT_FILE
)