from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. مسیرها
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "dual_beat_metadata.csv"
)


# =========================================================
# 2. خواندن Metadata
# =========================================================

df = pd.read_csv(
    METADATA_FILE
)


print(
    "Total Dual Beats:",
    len(df)
)


print(
    "\nBy split:"
)

print(
    df["split"].value_counts()
)


# =========================================================
# 3. انتخاب چند نمونه برای نمایش
#
# فعلاً:
# 2 تا Train
# 1 Validation
# 1 Test
# =========================================================

examples = []


train_examples = df[
    df["split"] == "train"
].head(2)

validation_examples = df[
    df["split"] == "validation"
].head(1)

test_examples = df[
    df["split"] == "test"
].head(1)


examples.extend(
    train_examples.to_dict(
        "records"
    )
)

examples.extend(
    validation_examples.to_dict(
        "records"
    )
)

examples.extend(
    test_examples.to_dict(
        "records"
    )
)


# =========================================================
# 4. نمایش Dual Beatها
# =========================================================

for example in examples:

    person_id = example[
        "person_id"
    ]

    record_id = example[
        "record_id"
    ]

    split_name = example[
        "split"
    ]

    dual_index = example[
        "dual_index"
    ]

    file_path = (
        PROJECT_DIR
        / example["file_path"]
    )


    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    dual_beat = np.load(
        file_path
    )


    print("\n------------------------------")

    print(
        "Person:",
        person_id
    )

    print(
        "Record:",
        record_id
    )

    print(
        "Split:",
        split_name
    )

    print(
        "Dual index:",
        dual_index
    )

    print(
        "Shape:",
        dual_beat.shape
    )

    print(
        "Minimum:",
        np.min(dual_beat)
    )

    print(
        "Maximum:",
        np.max(dual_beat)
    )


    # -----------------------------------------------------
    # هر Beat = 300 Sample
    # کل Dual = 600 Sample
    # -----------------------------------------------------

    beat_length = (
        len(dual_beat)
        // 2
    )


    # -----------------------------------------------------
    # محور Sample
    # -----------------------------------------------------

    samples = np.arange(
        len(dual_beat)
    )


    # -----------------------------------------------------
    # رسم
    # -----------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )


    plt.plot(
        samples,
        dual_beat
    )


    # مرز بین Beat 1 و Beat 2
    plt.axvline(
        x=beat_length,
        linestyle="--",
        label="Beat 1 / Beat 2 boundary"
    )


    plt.title(
        f"{person_id} - {record_id}"
        f" - Dual {dual_index}"
        f" - {split_name}"
    )


    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Normalized Amplitude"
    )


    plt.legend()

    plt.tight_layout()

    plt.show()