نسخه PostgreSQL با حداقل تغییر در enroll_user.py و verify_unlock.py

مهم:
- منطق ECG، preprocessing، Dual-Beat، image generation، model و authentication تغییر داده نشده است.
- فقط محل ذخیره/خواندن User Template از users.csv + .npy به PostgreSQL منتقل شده است.
- ابتدا PostgreSQL را نصب و یک database به نام ecg_smart_lock بسازید.
- فایل .env.example را به .env تغییر نام دهید و رمز PostgreSQL را وارد کنید.
- سپس:
    python src/init_database.py
- بعد:
    python src/enroll_user.py
    python src/verify_unlock.py

برای کاربران قدیمی، باید یک migration جداگانه از users.csv/.npy به PostgreSQL انجام شود.
