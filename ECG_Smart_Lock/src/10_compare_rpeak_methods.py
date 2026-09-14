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



methods = [
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "elgendi2010",
    "engzeemod2012",
]


records_to_check = [
    "rec_1",
    "rec_2",
    "rec_3",
]


for record_id in records_to_check:

    print("\n")
    print("========================================")
    print("Person_14", record_id)
    print("========================================")


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



    bio_signal = biometric_filter(
        raw_ecg,
        fs
    )

    bio_signal = (
        bio_signal
        - np.mean(bio_signal)
    ) / np.std(bio_signal)


    time = (
        np.arange(
            len(raw_ecg)
        )
        / fs
    )



    for method in methods:

        try:

            cleaned_for_detection = nk.ecg_clean(
                raw_ecg,
                sampling_rate=fs,
                method=method
            )


            signals, info = nk.ecg_peaks(
                cleaned_for_detection,
                sampling_rate=fs,
                method=method,
                correct_artifacts=False
            )

            r_peaks = info[
                "ECG_R_Peaks"
            ]



            duration = (
                len(raw_ecg)
                / fs
            )

            bpm = (
                len(r_peaks)
                / duration
            ) * 60


            if len(r_peaks) > 1:

                rr = np.diff(
                    r_peaks
                ) / fs

                median_rr = np.median(
                    rr
                )

            else:

                median_rr = np.nan


            print(
                f"{method:20s}"
                f" R={len(r_peaks):2d}"
                f" | BPM={bpm:6.1f}"
                f" | Median RR={median_rr}"
            )



            plt.figure(
                figsize=(16, 4)
            )

            plt.plot(
                time,
                bio_signal,
                label="Biometric ECG"
            )


            if len(r_peaks) > 0:

                plt.scatter(
                    r_peaks / fs,
                    bio_signal[r_peaks],
                    label=f"{method} R-peaks"
                )


            plt.title(
                f"Person 14 - {record_id} - {method}"
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


        except Exception as e:

            print(
                method,
                "ERROR:",
                e
            )