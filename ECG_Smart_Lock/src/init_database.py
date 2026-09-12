"""
init_database.py

یک‌بار اجرا کنید تا جدول‌های PostgreSQL ساخته شوند.
"""

from database import init_database


if __name__ == "__main__":
    print("در حال ساخت جدول‌های PostgreSQL ...")

    try:
        init_database()
        print("✓ جدول‌های PostgreSQL با موفقیت ساخته شدند.")
        print("  - users")
        print("  - ecg_templates")
    except Exception as e:
        print("❌ خطا در ساخت دیتابیس:")
        print(e)
