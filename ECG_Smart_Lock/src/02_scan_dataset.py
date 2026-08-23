from pathlib import Path
import pandas as pd
import wfdb


# -------------------------------
# 1. مسیر اصلی پروژه
# -------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "ecgiddb"
    / "ecg-id-database-1.0.0"
)


# -------------------------------
# 2. لیست برای ذخیره اطلاعات
# -------------------------------

records_info = []


# -------------------------------
# 3. پیدا کردن تمام Personها
# -------------------------------

person_folders = sorted(DATASET_DIR.glob("Person_*"))

print("Number of persons:", len(person_folders))


# -------------------------------
# 4. بررسی تک تک افراد
# -------------------------------

for person_folder in person_folders:

    person_id = person_folder.name

    # فقط فایل‌های .hea را پیدا می‌کنیم
    header_files = sorted(person_folder.glob("rec_*.hea"))

    print(person_id, "→", len(header_files), "records")

    for header_file in header_files:

        record_id = header_file.stem

        # wfdb پسوند .hea نمی‌خواهد
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


# -------------------------------
# 5. تبدیل اطلاعات به جدول
# -------------------------------

df = pd.DataFrame(records_info)


# -------------------------------
# 6. نمایش چند سطر اول
# -------------------------------

print("\nFirst records:")
print(df.head())


# -------------------------------
# 7. آمار کلی
# -------------------------------

print("\nTotal persons:")
print(df["person_id"].nunique())

print("\nTotal records:")
print(len(df))


# -------------------------------
# 8. تعداد Record هر فرد
# -------------------------------

records_per_person = (
    df.groupby("person_id")
    .size()
    .reset_index(name="number_of_records")
)

print("\nRecords per person:")
print(records_per_person)


# -------------------------------
# 9. ذخیره CSV
# -------------------------------

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