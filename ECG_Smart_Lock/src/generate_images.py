"""
generate_images.py

تبدیل داده‌های ECG جمع‌آوری‌شده با MAX30003 (برای همه‌ی افراد)
به تصاویر Dual-Beat با همان فرمت تصاویر ECG-ID.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import resample
from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# 1. اتصال به preprocessing.py
# =========================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import (
    biometric_filter,
    detect_and_refine_qrs,
    extract_beats
)


# =========================================================
# 2. تنظیمات
# =========================================================

# داده MAX30003 با نرخ 128Hz جمع‌آوری شده
ACTUAL_FS = 128

# نرخ نمونه‌برداری مورد استفاده در pipeline اصلی ECG-ID
TARGET_FS = 500

# اندازه تصویر
IMAGE_SIZE = 112
DPI = 100

# Dual-Beat باید دقیقاً 600 نمونه داشته باشد
X_MIN = 0
X_MAX = 599


# =========================================================
# 3. مسیرهای پروژه
# =========================================================

PROJECT_DIR = SRC_DIR.parent

# =========================================================
# مسیر ریشه‌ی داده‌ها - شامل پوشه‌ی هر فرد
#
# ساختار مورد انتظار:
#   data/max30003/<PERSON_ID>/rec_*.csv
#
# مطابق با کدی که برای جمع‌آوری داده نوشتی، PERSON_ID می‌تواند
# هر چیزی باشد که موقع اجرای اسکریپت جمع‌آوری تایپ کرده‌ای
# (مثلا "100", "101", "Person_01", ...)
# =========================================================

DATA_ROOT = (
    SRC_DIR
    / "MAX30003_ECG_128SPS.ino"
    / "data"
    / "max30003"
)

# اگر مسیر بالا با ساختار واقعی پروژه‌ات یکی نیست،
# فقط همین یک خط را با مسیر درست عوض کن. مثلا:
# DATA_ROOT = PROJECT_DIR / "data" / "max30003"


# ---------------------------------------------------------
# محل ذخیره تصاویر (به تفکیک هر فرد)
# ---------------------------------------------------------

IMAGE_OUTPUT_ROOT = (
    PROJECT_DIR
    / "outputs"
    / "images_new_data"
)


# ---------------------------------------------------------
# فایل مقیاس Y مربوط به ECG-ID
# ---------------------------------------------------------

SCALE_FILE = (
    PROJECT_DIR
    / "results"
    / "image_scale.txt"
)


# ---------------------------------------------------------
# فایل metadata نهایی (همه‌ی افراد در یک فایل)
# ---------------------------------------------------------

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "new_data_image_metadata.csv"
)


# =========================================================
# 4. بررسی مسیرها
# =========================================================

print("=" * 60)
print("MAX30003 ECG → Dual-Beat Images (همه‌ی افراد)")
print("=" * 60)

print(f"\nData root directory:")
print(DATA_ROOT)

print(f"\nOutput root directory:")
print(IMAGE_OUTPUT_ROOT)

print(f"\nScale file:")
print(SCALE_FILE)


if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"\nپوشه‌ی ریشه‌ی داده پیدا نشد:\n{DATA_ROOT}"
    )


if not SCALE_FILE.exists():
    raise FileNotFoundError(
        f"\nفایل image_scale.txt پیدا نشد:\n{SCALE_FILE}"
    )


# =========================================================
# 5. خواندن مقیاس Y
# =========================================================

def load_y_scale():

    with open(SCALE_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line.startswith("suggested_limit="):

                limit = float(
                    line.split("=", 1)[1]
                )

                return -limit, limit

    raise ValueError(
        "suggested_limit در image_scale.txt پیدا نشد."
    )


Y_MIN, Y_MAX = load_y_scale()

print(
    f"\nUsing ECG-ID Y-scale: "
    f"[{Y_MIN}, {Y_MAX}]"
)


# =========================================================
# 6. ساخت تصویر Dual-Beat
# =========================================================

def save_beat_image(dual_beat, image_path):

    fig = plt.figure(
        figsize=(
            IMAGE_SIZE / DPI,
            IMAGE_SIZE / DPI
        ),
        dpi=DPI
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    x = np.arange(len(dual_beat))

    ax.plot(
        x,
        dual_beat,
        color="black",
        linewidth=1.0
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.savefig(
        image_path,
        dpi=DPI,
        facecolor="white",
        edgecolor="white",
        pad_inches=0
    )

    plt.close(fig)

    image = Image.open(image_path).convert("L")

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image.save(image_path)


# =========================================================
# 7. پیدا کردن پوشه‌ی همه‌ی افراد (نه فقط یک نفر ثابت)
# =========================================================

person_folders = sorted(
    [p for p in DATA_ROOT.iterdir() if p.is_dir()]
)

print(f"\nFound {len(person_folders)} person folders:")
for p in person_folders:
    print(f"  - {p.name}")

if len(person_folders) == 0:
    raise FileNotFoundError(
        f"\nهیچ پوشه‌ی فردی داخل این مسیر پیدا نشد:\n{DATA_ROOT}"
    )


# =========================================================
# 8. پردازش تمام افراد و تمام فایل‌ها
# =========================================================

metadata = []
total_images = 0

for person_folder in person_folders:

    person_id = person_folder.name

    csv_files = sorted(person_folder.glob("rec_*.csv"))
    csv_files = [f for f in csv_files if f.suffix.lower() == ".csv"]

    print("\n" + "=" * 60)
    print(f"Person: {person_id}  ({len(csv_files)} files)")
    print("=" * 60)

    if len(csv_files) == 0:
        print("SKIP: هیچ فایل rec_*.csv پیدا نشد.")
        continue

    person_output_dir = IMAGE_OUTPUT_ROOT / person_id
    person_output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:

        record_id = csv_file.stem

        print("\n" + "-" * 60)
        print(f"Processing: {person_id} / {record_id}")
        print("-" * 60)

        # -----------------------------------------------------
        # خواندن CSV
        # -----------------------------------------------------

        df = pd.read_csv(csv_file)

        if "ecg" not in df.columns:
            print(f"SKIP: ستون 'ecg' در {csv_file.name} وجود ندارد.")
            print(f"Columns found: {list(df.columns)}")
            continue

        raw_ecg = df["ecg"].values.astype(float)

        if len(raw_ecg) == 0:
            print(f"SKIP: {record_id} خالی است.")
            continue

        print(f"Raw samples: {len(raw_ecg)}")
        print(f"Actual sampling rate: {ACTUAL_FS} Hz")

        # -----------------------------------------------------
        # Resample: 128Hz → 500Hz
        # -----------------------------------------------------

        n_samples_target = int(len(raw_ecg) * TARGET_FS / ACTUAL_FS)

        print(f"Resampling: {len(raw_ecg)} → {n_samples_target} samples")

        resampled_ecg = resample(raw_ecg, n_samples_target)
        fs = TARGET_FS

        # -----------------------------------------------------
        # فیلتر
        # -----------------------------------------------------

        try:
            filtered = biometric_filter(resampled_ecg, fs)
        except Exception as e:
            print(f"SKIP: filtering failed: {e}")
            continue

        # -----------------------------------------------------
        # Z-score normalization
        # -----------------------------------------------------

        mean_value = np.mean(filtered)
        std_value = np.std(filtered)

        if std_value == 0:
            print("SKIP: zero standard deviation.")
            continue

        biometric_ecg = (filtered - mean_value) / std_value

        # -----------------------------------------------------
        # QRS Detection + Refinement
        # -----------------------------------------------------

        try:
            anchors = detect_and_refine_qrs(resampled_ecg, fs, biometric_ecg)
        except Exception as e:
            print(f"SKIP: QRS detection failed: {e}")
            continue

        print(f"Detected R-peaks: {len(anchors)}")

        if len(anchors) < 2:
            print("SKIP: not enough R-peaks.")
            continue

        # -----------------------------------------------------
        # استخراج Beatها
        # -----------------------------------------------------

        beats = extract_beats(biometric_ecg, anchors, fs)

        print(f"Extracted beats: {len(beats)}")

        if len(beats) < 2:
            print("SKIP: not enough beats.")
            continue

        print(f"Beat shape: {beats.shape}")

        if beats.shape[1] != 300:
            print(f"WARNING: expected beat length 300, got {beats.shape[1]}")

        # -----------------------------------------------------
        # ساخت Dual-Beat
        # -----------------------------------------------------

        dual_count = 0

        for i in range(len(beats) - 1):

            beat_1 = beats[i]
            beat_2 = beats[i + 1]

            dual_beat = np.concatenate([beat_1, beat_2])

            if len(dual_beat) != 600:
                print(f"Skipping dual beat {i}: length={len(dual_beat)}")
                continue

            dual_count += 1
            total_images += 1

            image_name = f"{person_id}_{record_id}_dual_{dual_count:03d}.png"
            image_path = person_output_dir / image_name

            save_beat_image(dual_beat, image_path)

            metadata.append({
                "person_id": person_id,
                "record_id": record_id,
                "dual_id": dual_count,
                "split": "finetune",
                "image_path": str(image_path.relative_to(PROJECT_DIR)),
            })

        print(f"\n{person_id} / {record_id} finished:")
        print(f"  R-peaks       = {len(anchors)}")
        print(f"  Beats         = {len(beats)}")
        print(f"  Dual images   = {dual_count}")


# =========================================================
# 9. ذخیره Metadata
# =========================================================

if metadata:

    metadata_df = pd.DataFrame(metadata)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(METADATA_FILE, index=False)

    print("\n" + "=" * 60)
    print("Finished successfully!")
    print(f"Total persons processed: {len(person_folders)}")
    print(f"Total images: {total_images}")
    print(f"Images saved in:\n{IMAGE_OUTPUT_ROOT}")
    print(f"Metadata saved in:\n{METADATA_FILE}")
    print("=" * 60)

else:
    print("\n" + "=" * 60)
    print("هیچ تصویری ساخته نشد.")
    print("لطفاً خروجی R-peaks و Beats را برای هر فرد بررسی کن.")
    print("=" * 60)