import csv
from pathlib import Path

import numpy as np

from database import init_database, upsert_user_with_template


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

USERS_DIR = PROJECT_DIR / "results" / "user_database"
USERS_FILE = USERS_DIR / "users.csv"

MODEL_VERSION = "finetuned_model"


def main():
    init_database()

    if not USERS_FILE.exists():
        print("❌ users.csv پیدا نشد:")
        print(USERS_FILE)
        return

    with open(
        USERS_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("❌ users.csv خالی است.")
        return

    success_count = 0

    for row in rows:
        user_name = (
            row.get("user_name")
            or row.get("name")
            or ""
        ).strip()

        if not user_name:
            print("⚠️ نام کاربر خالی است؛ رد شد.")
            continue

        embedding_path = str(
            row.get("embedding_path") or ""
        ).strip()

        if embedding_path and embedding_path.lower() != "nan":
            path = Path(embedding_path)

            if not path.is_absolute():
                path = PROJECT_DIR / path
        else:
            path = USERS_DIR / f"{user_name}.npy"

        if not path.exists():
            path = USERS_DIR / f"{user_name}.npy"

        if not path.exists():
            print(
                f"⚠️ Template کاربر '{user_name}' پیدا نشد:"
            )
            print(path)
            continue

        try:
            embedding = np.load(path)
            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            num_beats = int(
                row.get("num_beats") or 0
            )

            upsert_user_with_template(
                user_name=user_name,
                num_beats=num_beats,
                embedding=embedding,
                model_version=MODEL_VERSION,
            )

            print(
                f"✓ {user_name} -> PostgreSQL"
            )
            success_count += 1

        except Exception as e:
            print(
                f"❌ خطا برای کاربر '{user_name}':"
            )
            print(e)

    print()
    print("=" * 60)
    print(
        f"تعداد کاربران منتقل‌شده: {success_count}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
