from pathlib import Path
import pandas as pd
import re


PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_DIR
    / "results"
    / "dataset_metadata.csv"
)

RESULTS_DIR = PROJECT_DIR / "results"


df = pd.read_csv(METADATA_FILE)

print("Total records in metadata:", len(df))
print("Total persons in metadata:", df["person_id"].nunique())


def get_record_number(record_id):

    match = re.search(r"(\d+)$", record_id)

    if match:
        return int(match.group(1))

    return -1


df["record_number"] = df["record_id"].apply(get_record_number)



record_counts = (
    df.groupby("person_id")
    .size()
)



eligible_persons = record_counts[
    record_counts >= 3
].index


print(
    "Persons with at least 3 records:",
    len(eligible_persons)
)


df_eligible = df[
    df["person_id"].isin(eligible_persons)
].copy()



df_eligible["split"] = ""



for person_id in eligible_persons:

    person_records = df_eligible[
        df_eligible["person_id"] == person_id
    ].copy()

    person_records = person_records.sort_values(
        "record_number"
    )

    indices = person_records.index.tolist()

    train_indices = indices[:-2]

    val_index = indices[-2]

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