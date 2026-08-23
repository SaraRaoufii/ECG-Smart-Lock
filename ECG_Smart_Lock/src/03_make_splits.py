from pathlib import Path
import pandas as pd
import re


# --------------------------------
# 1. مسیر پروژه
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "dataset_metadata.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"


# --------------------------------
# 2. خواندن Metadata
# --------------------------------

df = pd.read_csv(METADATA_FILE)

print("Total records in metadata:", len(df))
print("Total persons in metadata:", df["person_id"].nunique())


# --------------------------------
# 3. تابع گرفتن شماره واقعی Record
# --------------------------------

def get_record_number(record_id):

    match = re.search(r"(\d+)$", record_id)

    if match:
        return int(match.group(1))

    return -1


df["record_number"] = df["record_id"].apply(get_record_number)


# --------------------------------
# 4. شمارش Record هر فرد
# --------------------------------

record_counts = (
    df.groupby("person_id")
    .size()
)


# --------------------------------
# 5. انتخاب افراد دارای حداقل 3 Record
# --------------------------------

eligible_persons = record_counts[
    record_counts >= 3
].index


print(
    "Persons with at least 3 records:",
    len(eligible_persons)
)


# فقط همین افراد
df_eligible = df[
    df["person_id"].isin(eligible_persons)
].copy()


# --------------------------------
# 6. ستون split
# --------------------------------

df_eligible["split"] = ""


# --------------------------------
# 7. تقسیم Recordهای هر فرد
# --------------------------------

for person_id in eligible_persons:

    person_records = df_eligible[
        df_eligible["person_id"] == person_id
    ].copy()

    # مرتب‌سازی بر اساس شماره Record
    person_records = person_records.sort_values(
        "record_number"
    )

    indices = person_records.index.tolist()

    # همه به جز دو Record آخر → Train
    train_indices = indices[:-2]

    # یکی مانده به آخر → Validation
    val_index = indices[-2]

    # آخرین Record → Test
    test_index = indices[-1]

    df_eligible.loc[
        train_indices,
        "split"
    ] = "train"

    df_eligible.loc[
        val_index,
        "split"
    ] = "validation"

    df_eligible.loc[
        test_index,
        "split"
    ] = "test"


# --------------------------------
# 8. نمایش نتیجه
# --------------------------------

print("\nSplit counts:")

print(
    df_eligible["split"].value_counts()
)


print("\nPersons in each split:")

for split_name in [
    "train",
    "validation",
    "test"
]:

    subset = df_eligible[
        df_eligible["split"] == split_name
    ]

    print(
        split_name,
        "records:",
        len(subset),
        "| persons:",
        subset["person_id"].nunique()
    )


# --------------------------------
# 9. نمونه Person_01
# --------------------------------

print("\nExample - Person_01:")

print(
    df_eligible[
        df_eligible["person_id"] == "Person_01"
    ][
        [
            "person_id",
            "record_id",
            "record_number",
            "split"
        ]
    ]
)


# --------------------------------
# 10. ذخیره Split اصلی
# --------------------------------

output_file = (
    RESULTS_DIR
    / "dataset_split.csv"
)

df_eligible.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)