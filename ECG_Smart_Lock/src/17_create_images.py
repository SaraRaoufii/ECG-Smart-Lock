from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


# =========================================================
# 1. مسیرها
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "dual_beat_metadata.csv"
)

IMAGE_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "images"
    / "112x112"
)

IMAGE_METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "image_metadata.csv"
)


# =========================================================
# 2. تنظیمات ثابت تصویر
# =========================================================

IMAGE_SIZE = 112

DPI = 100

Y_MIN = -7.5
Y_MAX = 7.5

X_MIN = 0
X_MAX = 599


# =========================================================
# 3. خواندن Metadata
# =========================================================

df = pd.read_csv(
    METADATA_FILE
)


print(
    "Total Dual Beats:",
    len(df)
)


# =========================================================
# 4. پوشه خروجی
# =========================================================

IMAGE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 5. لیست Metadata تصاویر
# =========================================================

image_metadata = []


# =========================================================
# 6. تبدیل Dual Beat به تصویر
# =========================================================

for index, row in df.iterrows():

    person_id = row[
        "person_id"
    ]

    record_id = row[
        "record_id"
    ]

    split_name = row[
        "split"
    ]

    dual_index = int(
        row[
            "dual_index"
        ]
    )


    # -----------------------------------------------------
    # مسیر فایل NPY
    # -----------------------------------------------------

    dual_path = (
        PROJECT_DIR
        / row["file_path"]
    )


    # -----------------------------------------------------
    # Load Dual Beat
    # -----------------------------------------------------

    dual_beat = np.load(
        dual_path
    )


    # کنترل طول
    if len(dual_beat) != 600:

        print(
            "SKIP - wrong length:",
            dual_path
        )

        continue


    # -----------------------------------------------------
    # پوشه Split / Person
    # -----------------------------------------------------

    person_output_dir = (
        IMAGE_OUTPUT_DIR
        / split_name
        / person_id
    )

    person_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # نام تصویر
    # -----------------------------------------------------

    image_name = (
        f"{person_id}_"
        f"{record_id}_"
        f"dual_{dual_index:03d}.png"
    )


    image_path = (
        person_output_dir
        / image_name
    )


    # -----------------------------------------------------
    # ساخت Figure دقیقاً 112 × 112 pixel
    #
    # 112 pixel / 100 dpi = 1.12 inch
    # -----------------------------------------------------

    fig = plt.figure(
        figsize=(
            IMAGE_SIZE / DPI,
            IMAGE_SIZE / DPI
        ),
        dpi=DPI
    )


    # کل Figure متعلق به ECG باشد
    ax = fig.add_axes(
        [0, 0, 1, 1]
    )


    # -----------------------------------------------------
    # ECG
    # -----------------------------------------------------

    x = np.arange(
        len(dual_beat)
    )


    ax.plot(
        x,
        dual_beat,
        color="black",
        linewidth=1.0
    )


    # -----------------------------------------------------
    # Scale ثابت برای همه تصاویر
    # -----------------------------------------------------

    ax.set_xlim(
        X_MIN,
        X_MAX
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX
    )


    # -----------------------------------------------------
    # حذف تمام اجزای اضافی
    # -----------------------------------------------------

    ax.axis(
        "off"
    )


    # Background سفید
    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )


    # -----------------------------------------------------
    # ذخیره PNG
    # -----------------------------------------------------

    fig.savefig(
        image_path,
        dpi=DPI,
        facecolor="white",
        edgecolor="white",
        pad_inches=0
    )


    plt.close(
        fig
    )


    # -----------------------------------------------------
    # تبدیل قطعی به Grayscale
    # و کنترل سایز
    # -----------------------------------------------------

    image = Image.open(
        image_path
    )

    image = image.convert(
        "L"
    )

    # کنترل نهایی سایز
    if image.size != (
        IMAGE_SIZE,
        IMAGE_SIZE
    ):

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )


    image.save(
        image_path
    )


    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    image_metadata.append(
        {
            "person_id": person_id,
            "record_id": record_id,
            "split": split_name,

            "dual_index": dual_index,

            "image_width": (
                IMAGE_SIZE
            ),

            "image_height": (
                IMAGE_SIZE
            ),

            "image_mode": "L",

            "image_path": str(
                image_path.relative_to(
                    PROJECT_DIR
                )
            ),
        }
    )


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if (
        index + 1
    ) % 500 == 0:

        print(
            f"Created {index + 1}"
            f" / {len(df)} images"
        )


# =========================================================
# 7. Metadata تصاویر
# =========================================================

image_df = pd.DataFrame(
    image_metadata
)


image_df.to_csv(
    IMAGE_METADATA_FILE,
    index=False
)


# =========================================================
# 8. Summary
# =========================================================

print("\n")
print(
    "===================================="
)

print(
    "IMAGE CREATION SUMMARY"
)

print(
    "===================================="
)


print(
    "Created images:",
    len(image_df)
)


if len(image_df) > 0:

    print(
        "\nBy split:"
    )

    print(
        image_df[
            "split"
        ].value_counts()
    )


    print(
        "\nImage sizes:"
    )

    print(
        image_df[
            [
                "image_width",
                "image_height"
            ]
        ].value_counts()
    )


    print(
        "\nImage modes:"
    )

    print(
        image_df[
            "image_mode"
        ].value_counts()
    )


print(
    "\nSaved metadata:"
)

print(
    IMAGE_METADATA_FILE
)