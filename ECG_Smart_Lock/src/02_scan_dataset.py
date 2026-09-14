from pathlib import Path
import pandas as pd
import wfdb



PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "ecgiddb"
    / "ecg-id-database-1.0.0"
)


records_info = []



person_folders = sorted(DATASET_DIR.glob("Person_*"))

print("Number of persons:", len(person_folders))



for person_folder in person_folders:

    person_id = person_folder.name


    header_files = sorted(person_folder.glob("rec_*.hea"))

    print(person_id, "→", len(header_files), "records")

    for header_file in header_files:

        record_id = header_file.stem

        record_path = person_folder / record_id

        try:
            record = wfdb.rdrecord(str(record_path))

            records_info.append(
                {
                    "person_id": person_id,
                    "record_id": record_id,
                    "sampling_rate": record.fs,
                    "n_samples": record.sig_len,
                    "duration_seconds": record.sig_len / record.fs,
                    "n_channels": record.n_sig,
                }
            )

        except Exception as e:

            print("ERROR:", person_id, record_id, e)


df = pd.DataFrame(records_info)



print("\nFirst records:")
print(df.head())


print("\nTotal persons:")
print(df["person_id"].nunique())

print("\nTotal records:")
print(len(df))



records_per_person = (
    df.groupby("person_id")
    .size()
    .reset_index(name="number_of_records")
)

print("\nRecords per person:")
print(records_per_person)



results_dir = PROJECT_DIR / "results"

results_dir.mkdir(exist_ok=True)

df.to_csv(
    results_dir / "dataset_metadata.csv",
    index=False
)

records_per_person.to_csv(
    results_dir / "records_per_person.csv",
    index=False
)


print("\nSaved:")
print(results_dir / "dataset_metadata.csv")
print(results_dir / "records_per_person.csv")