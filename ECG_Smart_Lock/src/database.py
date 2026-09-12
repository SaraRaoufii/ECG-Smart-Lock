"""
database.py

لایه‌ی دسترسی PostgreSQL برای سامانه ECG Smart Lock.

ساختار:
    users
        id
        user_name
        num_beats
        created_at
        updated_at

    ecg_templates
        id
        user_id
        embedding
        embedding_dim
        model_version
        created_at
        updated_at

Embedding به صورت BYTEA ذخیره می‌شود و هنگام خواندن
به numpy.ndarray با dtype=float32 تبدیل می‌شود.
"""

import os
from datetime import datetime, timezone

import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()
from psycopg2 import sql


def get_database_config():
    """خواندن تنظیمات PostgreSQL از Environment Variables."""

    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "ecg_smart_lock"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }


def get_connection():
    """ایجاد اتصال به PostgreSQL."""

    return psycopg2.connect(**get_database_config())


def init_database():
    """ساخت جدول‌های مورد نیاز در صورت نبودن."""

    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        user_name VARCHAR(100) NOT NULL UNIQUE,
        num_beats INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_templates_table = """
    CREATE TABLE IF NOT EXISTS ecg_templates (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL UNIQUE
            REFERENCES users(id)
            ON DELETE CASCADE,
        embedding BYTEA NOT NULL,
        embedding_dim INTEGER NOT NULL,
        model_version VARCHAR(100) NOT NULL DEFAULT 'finetuned_model',
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_users_table)
            cur.execute(create_templates_table)
        conn.commit()


def _embedding_to_bytes(embedding):
    """تبدیل embedding به bytes مناسب BYTEA."""

    array = np.asarray(embedding, dtype=np.float32)

    if array.ndim != 1:
        raise ValueError(
            f"Embedding باید یک‌بعدی باشد. Shape = {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding شامل NaN یا Inf است.")

    if array.size == 0:
        raise ValueError("Embedding خالی است.")

    return array.tobytes(), int(array.shape[0])


def _bytes_to_embedding(data, dimension):
    """تبدیل BYTEA دیتابیس به numpy.ndarray."""

    if data is None:
        raise ValueError("Embedding در دیتابیس خالی است.")

    array = np.frombuffer(data, dtype=np.float32).copy()

    if len(array) != int(dimension):
        raise ValueError(
            f"ابعاد embedding با مقدار ذخیره‌شده سازگار نیست: "
            f"{len(array)} != {dimension}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding دیتابیس شامل NaN یا Inf است.")

    return array


def upsert_user_with_template(
    user_name,
    num_beats,
    embedding,
    model_version="finetuned_model",
):
    """
    ثبت کاربر و Template.

    اگر user_name از قبل وجود داشته باشد:
        اطلاعات کاربر و Template آن به‌روزرسانی می‌شوند.

    در نتیجه رفتار قبلی users.csv نیز حفظ می‌شود:
        کاربر جدید -> INSERT
        کاربر موجود -> UPDATE
    """

    embedding_bytes, embedding_dim = _embedding_to_bytes(embedding)

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO users (
                    user_name,
                    num_beats,
                    updated_at
                )
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_name)
                DO UPDATE SET
                    num_beats = EXCLUDED.num_beats,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
                """,
                (user_name.strip(), int(num_beats)),
            )

            user_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO ecg_templates (
                    user_id,
                    embedding,
                    embedding_dim,
                    model_version,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_dim = EXCLUDED.embedding_dim,
                    model_version = EXCLUDED.model_version,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (
                    user_id,
                    psycopg2.Binary(embedding_bytes),
                    embedding_dim,
                    model_version,
                ),
            )

        conn.commit()

    return True


def get_all_templates():
    """
    دریافت تمام Templateهای معتبر.

    خروجی:
        {
            "Sara": np.ndarray(...),
            "Ali": np.ndarray(...),
            ...
        }
    """

    templates = {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    u.user_name,
                    t.embedding,
                    t.embedding_dim
                FROM users AS u
                INNER JOIN ecg_templates AS t
                    ON t.user_id = u.id
                ORDER BY u.id;
                """
            )

            rows = cur.fetchall()

    for user_name, embedding_data, embedding_dim in rows:
        embedding = _bytes_to_embedding(
            embedding_data,
            embedding_dim,
        )
        templates[str(user_name).strip()] = embedding

    return templates


def get_user_template(user_name):
    """دریافت Template یک کاربر خاص."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.embedding,
                    t.embedding_dim
                FROM users AS u
                INNER JOIN ecg_templates AS t
                    ON t.user_id = u.id
                WHERE LOWER(u.user_name) = LOWER(%s);
                """,
                (user_name.strip(),),
            )

            row = cur.fetchone()

    if row is None:
        return None

    return _bytes_to_embedding(row[0], row[1])


def count_users():
    """تعداد کاربران دارای Template."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM users AS u
                INNER JOIN ecg_templates AS t
                    ON t.user_id = u.id;
                """
            )
            return int(cur.fetchone()[0])


def delete_user(user_name):
    """حذف کاربر و Template مرتبط."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM users
                WHERE LOWER(user_name) = LOWER(%s);
                """,
                (user_name.strip(),),
            )
            deleted = cur.rowcount

        conn.commit()

    return deleted > 0
