# PostgreSQL migration for ECG Smart Lock

## 1. Install packages

Inside the project virtual environment:

```bash
python -m pip install psycopg2-binary python-dotenv
```

## 2. Create PostgreSQL database

In pgAdmin or psql create:

```sql
CREATE DATABASE ecg_smart_lock;
```

## 3. Configure environment variables

Create `.env` in the project root:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ecg_smart_lock
POSTGRES_USER=postgres
POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD
```

If your PostgreSQL username/password are different, change them.

## 4. Important: load .env

At the very beginning of `database.py`, before `get_database_config()`,
you can add:

```python
from dotenv import load_dotenv
load_dotenv()
```

Then the `.env` file will be loaded automatically.

## 5. Initialize tables

From the folder containing `src`:

```bash
python src/init_database.py
```

Expected tables:

- users
- ecg_templates

## 6. If old users already exist

Run:

```bash
python src/migrate_csv_to_postgres.py
```

This transfers:

```text
results/user_database/users.csv
results/user_database/*.npy
```

to PostgreSQL.

## 7. New enrollment

The new `enroll_user.py` stores:

- user name
- number of embeddings/beats
- normalized 128-dimensional template
- model version
- timestamps

directly in PostgreSQL.

The ECG processing pipeline itself is unchanged.

## 8. Verification

The new `verify_unlock.py` reads all templates directly from PostgreSQL.
It no longer reads `users.csv` or `.npy`.

## 9. Data model

```text
users
--------------------------------
id
user_name
num_beats
created_at
updated_at

        1
        |
        | 1
        v

ecg_templates
--------------------------------
id
user_id
embedding
embedding_dim
model_version
created_at
updated_at
```

The `embedding` is stored as PostgreSQL `BYTEA`.

## 10. Existing live ECG files

`results/live_ecg.json` is NOT moved to PostgreSQL.

It is still used as a temporary communication file between
the Arduino recording process and the Flask web UI.
