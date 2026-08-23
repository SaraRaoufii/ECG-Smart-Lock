from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. مسیرها
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "dual_beat_metadata.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "image_scale.txt"
)


# =========================================================
# 2. خواندن Metadata
# =========================================================

df = pd.read_csv(
    METADATA_FILE
)


# =========================================================
# 3. فقط TRAIN
#
# Validation و Test برای تعیین Scale استفاده نمی‌شوند.
# =========================================================

train_df = df[
    df["split"] == "train"
].copy()


print(
    "Train Dual Beats:",
    len(train_df)
)


# =========================================================
# 4. خواندن تمام Dual Beatهای Train
# =========================================================

all_values = []


for _, row in train_df.iterrows():

    file_path = (
        PROJECT_DIR
        / row["file_path"]
    )


    dual_beat = np.load(
        file_path
    )


    all_values.append(
        dual_beat
    )


# تبدیل به یک آرایه بزرگ
all_values = np.concatenate(
    all_values
)


# =========================================================
# 5. آمار کلی
# =========================================================

minimum = np.min(
    all_values
)

maximum = np.max(
    all_values
)


p001 = np.percentile(
    all_values,
    0.1
)

p01 = np.percentile(
    all_values,
    1
)

p99 = np.percentile(
    all_values,
    99
)

p999 = np.percentile(
    all_values,
    99.9
)


print("\n==============================")
print("TRAIN AMPLITUDE STATISTICS")
print("==============================")


print(
    "Minimum:",
    minimum
)

print(
    "Maximum:",
    maximum
)

print(
    "0.1 percentile:",
    p001
)

print(
    "1 percentile:",
    p01
)

print(
    "99 percentile:",
    p99
)

print(
    "99.9 percentile:",
    p999
)


# =========================================================
# 6. قدر مطلق
# =========================================================

absolute_values = np.abs(
    all_values
)


abs_99 = np.percentile(
    absolute_values,
    99
)

abs_999 = np.percentile(
    absolute_values,
    99.9
)


print(
    "\nAbsolute 99 percentile:",
    abs_99
)

print(
    "Absolute 99.9 percentile:",
    abs_999
)


# =========================================================
# 7. پیشنهاد Scale متقارن
#
# فعلاً از 99.9 percentile قدر مطلق استفاده می‌کنیم.
# بعد از دیدن عدد، مقدار نهایی را Round می‌کنیم.
# =========================================================

suggested_limit = abs_999


print(
    "\nSuggested symmetric Y limit:"
)

print(
    -suggested_limit,
    "to",
    suggested_limit
)


# =========================================================
# 8. ذخیره خروجی
# =========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"minimum={minimum}\n"
    )

    f.write(
        f"maximum={maximum}\n"
    )

    f.write(
        f"percentile_0.1={p001}\n"
    )

    f.write(
        f"percentile_1={p01}\n"
    )

    f.write(
        f"percentile_99={p99}\n"
    )

    f.write(
        f"percentile_99.9={p999}\n"
    )

    f.write(
        f"absolute_99={abs_99}\n"
    )

    f.write(
        f"absolute_99.9={abs_999}\n"
    )

    f.write(
        f"suggested_limit={suggested_limit}\n"
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)