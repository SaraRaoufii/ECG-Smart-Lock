
"""
verify_unlock.py

دریافت سیگنال ECG از MAX30003،
ساخت Embeddingهای ECG،
مقایسه با Template کاربران ثبت‌شده،
و تصمیم‌گیری برای احراز هویت.

ویژگی‌های این نسخه:

    - ضبط 35 ثانیه ECG
    - حذف 10 ثانیه ابتدایی فقط در signal_to_embeddings()
    - استفاده از همان preprocessing مرحله Enrollment
    - ساخت Query Embedding از میانگین Embeddingهای حاصل
    - L2 Normalization برای Query
    - مقایسه با Template تمام کاربران
    - تحلیل تک‌تک Embeddingها
    - Threshold ثابت ذخیره‌شده در unlock_threshold.txt
    - Margin Check برای جلوگیری از پذیرش نتایج نامطمئن
    - فعلاً بدون ارسال فرمان رله
"""

import sys

from pathlib import Path

import numpy as np
import torch


# =========================================================
# مسیر پروژه
# =========================================================

# verify_unlock.py داخل src قرار دارد.
# بنابراین parent = src
# و parent.parent = ریشه اصلی پروژه ECG_Smart_Lock

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# =========================================================
# Import model
# =========================================================

from model import get_model

from database import init_database, get_all_templates


# =========================================================
# Import preprocessing / recording
# =========================================================

from enroll_user import (
    record_ecg,
    signal_to_embeddings,
    PORT,
    BAUDRATE,
    MODEL_PATH,
    DEVICE,
)


# =========================================================
# مسیرهای اصلی
# =========================================================

RESULTS_DIR = PROJECT_DIR / "results"

# Template کاربران در PostgreSQL نگهداری می‌شود.

# محل واقعی Threshold
THRESHOLD_FILE = RESULTS_DIR / "unlock_threshold.txt"


# =========================================================
# تنظیمات ضبط
# =========================================================

WARMUP_SECONDS = 10

USEFUL_DURATION_SECONDS = 25

VERIFY_DURATION_SECONDS = (
    WARMUP_SECONDS
    + USEFUL_DURATION_SECONDS
)

SAMPLING_RATE = 128


# =========================================================
# تنظیمات کیفیت
# =========================================================

MIN_VERIFY_EMBEDDINGS = 10


# =========================================================
# Threshold
# =========================================================

DEFAULT_THRESHOLD = 0.9126


# =========================================================
# Margin
# =========================================================

MIN_DISTANCE_MARGIN = 0.05


# =========================================================
# تعداد Embeddingهایی که چاپ می‌شوند
# =========================================================

MAX_EMBEDDINGS_TO_PRINT = 25


# =========================================================
# خواندن Threshold
# =========================================================

def load_unlock_threshold():

    if not THRESHOLD_FILE.exists():

        print()

        print(
            "⚠️ فایل unlock_threshold.txt پیدا نشد."
        )

        print(
            "مسیر مورد انتظار:"
        )

        print(
            THRESHOLD_FILE
        )

        print(
            f"از Threshold پیش‌فرض "
            f"{DEFAULT_THRESHOLD:.4f} استفاده می‌شود."
        )

        return DEFAULT_THRESHOLD

    try:

        with open(
            THRESHOLD_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read().strip()

        # -------------------------------------------------
        # حالت اول:
        # threshold=0.9126
        # -------------------------------------------------

        for line in content.splitlines():

            line = line.strip()

            if line.startswith("threshold="):

                try:

                    threshold = float(
                        line.split("=", 1)[1].strip()
                    )

                    if (
                        np.isfinite(threshold)
                        and threshold > 0
                    ):

                        return threshold

                except ValueError:

                    pass

        # -------------------------------------------------
        # حالت دوم:
        # اگر فایل فقط عدد داشته باشد
        #
        # مثلاً:
        # 0.9126
        # -------------------------------------------------

        try:

            threshold = float(
                content
            )

            if (
                np.isfinite(threshold)
                and threshold > 0
            ):

                return threshold

        except ValueError:

            pass

    except Exception as e:

        print()

        print(
            "⚠️ خطا در خواندن فایل Threshold:"
        )

        print(e)

    print()

    print(
        "⚠️ مقدار Threshold معتبر نیست."
    )

    print(
        f"از مقدار پیش‌فرض "
        f"{DEFAULT_THRESHOLD:.4f} استفاده می‌شود."
    )

    return DEFAULT_THRESHOLD


UNLOCK_THRESHOLD = load_unlock_threshold()


# =========================================================
# Normalize embedding
# =========================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if (
        not np.isfinite(norm)
        or norm == 0
    ):

        raise ValueError(
            "Embedding قابل Normalize نیست."
        )

    return (
        embedding / norm
    ).astype(np.float32)


# =========================================================
# Load user database
# =========================================================

def load_database():
    print()

    print(
        "در حال بارگذاری دیتابیس کاربران از PostgreSQL..."
    )

    try:
        init_database()
        templates = get_all_templates()
    except Exception as e:
        print()
        print(
            "❌ خطا در خواندن PostgreSQL:"
        )
        print(e)
        raise

    if not templates:
        raise RuntimeError(
            "هیچ Template معتبری در PostgreSQL پیدا نشد."
        )

    normalized_templates = {}

    for user_name, embedding in templates.items():

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        if embedding.ndim != 1:
            print()
            print(
                f"⚠️ Embedding کاربر "
                f"'{user_name}' معتبر نیست."
            )
            continue

        if embedding.shape[0] != 128:
            print()
            print(
                f"⚠️ هشدار: Embedding کاربر "
                f"'{user_name}' دارای "
                f"{embedding.shape[0]} بعد است."
            )

        if not np.all(
            np.isfinite(embedding)
        ):
            print()
            print(
                f"⚠️ Embedding کاربر "
                f"'{user_name}' شامل "
                f"مقادیر نامعتبر است."
            )
            continue

        try:
            embedding = normalize_embedding(
                embedding
            )
        except ValueError as e:
            print()
            print(
                f"⚠️ Template کاربر "
                f"'{user_name}' نامعتبر است:"
            )
            print(e)
            continue

        normalized_templates[user_name] = embedding

        print(
            f"  ✓ {user_name:<12} <- PostgreSQL"
        )

    if len(normalized_templates) == 0:
        raise RuntimeError(
            "هیچ Template معتبر "
            "در PostgreSQL پیدا نشد."
        )

    print()
    print(
        f"✓ تعداد Templateهای معتبر: "
        f"{len(normalized_templates)}"
    )

    return normalized_templates


# =========================================================
# محاسبه فاصله Euclidean
# =========================================================

def calculate_distance(
    embedding1,
    embedding2
):

    return float(
        np.linalg.norm(
            embedding1
            - embedding2
        )
    )


# =========================================================
# تحلیل تک‌تک Embeddingها
# =========================================================

def analyze_individual_embeddings(
    embeddings,
    templates
):

    print()

    print("=" * 80)

    print(
        "ANALYSIS OF INDIVIDUAL EMBEDDINGS"
    )

    print("=" * 80)

    total_embeddings = len(
        embeddings
    )

    print(
        f"تعداد کل Embeddingها: "
        f"{total_embeddings}"
    )

    # -----------------------------------------------------
    # فاصله‌های هر کاربر
    # -----------------------------------------------------

    user_distances = {

        user_name: []

        for user_name in templates

    }

    # -----------------------------------------------------
    # تعداد دفعات نزدیک‌ترین بودن
    # -----------------------------------------------------

    closest_counts = {

        user_name: 0

        for user_name in templates

    }

    # -----------------------------------------------------
    # تعداد Embeddingهایی که هیچ
    # کاربری زیر Threshold نیست
    # -----------------------------------------------------

    no_match_count = 0

    # -----------------------------------------------------
    # تعداد مورد چاپ
    # -----------------------------------------------------

    if MAX_EMBEDDINGS_TO_PRINT is None:

        print_count = total_embeddings

    else:

        print_count = min(
            total_embeddings,
            MAX_EMBEDDINGS_TO_PRINT
        )

    # -----------------------------------------------------
    # بررسی Embeddingها
    # -----------------------------------------------------

    for i, embedding in enumerate(
        embeddings
    ):

        # Query embedding را Normalize می‌کنیم

        embedding = normalize_embedding(
            embedding
        )

        distances = []

        for (
            user_name,
            template
        ) in templates.items():

            distance = calculate_distance(
                embedding,
                template
            )

            distances.append(
                (
                    user_name,
                    distance
                )
            )

            user_distances[
                user_name
            ].append(
                distance
            )

        # -------------------------------------------------
        # مرتب‌سازی
        # -------------------------------------------------

        distances.sort(
            key=lambda item: item[1]
        )

        closest_user = distances[0][0]

        closest_distance = distances[0][1]

        closest_counts[
            closest_user
        ] += 1

        # -------------------------------------------------
        # Threshold
        # -------------------------------------------------

        if (
            closest_distance
            >= UNLOCK_THRESHOLD
        ):

            no_match_count += 1

        # -------------------------------------------------
        # چاپ
        # -------------------------------------------------

        if i < print_count:

            print()

            print(
                f"Embedding #{i + 1}"
            )

            print(
                "-" * 50
            )

            for (
                user_name,
                distance
            ) in distances:

                print(
                    f"  {user_name:<12} "
                    f"{distance:.4f}"
                )

            print(
                f"  --> نزدیک‌ترین: "
                f"{closest_user} "
                f"({closest_distance:.4f})"
            )

    # =====================================================
    # خلاصه
    # =====================================================

    print()

    print("=" * 80)

    print(
        "INDIVIDUAL EMBEDDING SUMMARY"
    )

    print("=" * 80)

    for user_name in templates:

        distances = np.asarray(
            user_distances[user_name],
            dtype=np.float32
        )

        print()

        print(
            f"کاربر: {user_name}"
        )

        print(
            f"  Mean   = "
            f"{np.mean(distances):.4f}"
        )

        print(
            f"  Median = "
            f"{np.median(distances):.4f}"
        )

        print(
            f"  Min    = "
            f"{np.min(distances):.4f}"
        )

        print(
            f"  Max    = "
            f"{np.max(distances):.4f}"
        )

        print(
            f"  Std    = "
            f"{np.std(distances):.4f}"
        )

    # =====================================================
    # نزدیک‌ترین بودن
    # =====================================================

    print()

    print(
        "-" * 80
    )

    print(
        "تعداد دفعاتی که هر کاربر "
        "نزدیک‌ترین Template بوده:"
    )

    for (
        user_name,
        count
    ) in closest_counts.items():

        percentage = (
            count
            / total_embeddings
            * 100
        )

        print(
            f"  {user_name:<12}: "
            f"{count:>3} / "
            f"{total_embeddings} "
            f"({percentage:.1f}%)"
        )

    print()

    print(
        f"Embeddingهایی که حتی "
        f"نزدیک‌ترین کاربرشان "
        f"زیر Threshold نیست: "
        f"{no_match_count}"
    )

    return (
        user_distances,
        closest_counts
    )


# =========================================================
# Main
# =========================================================

def main():

    print()

    print("=" * 60)

    print(
        "        ECG SMART LOCK - VERIFY"
    )

    print("=" * 60)

    print(
        f"نرخ نمونه‌برداری: "
        f"{SAMPLING_RATE} Hz"
    )

    print(
        f"Warm-up: "
        f"{WARMUP_SECONDS} ثانیه"
    )

    print(
        f"مدت مفید ECG: "
        f"{USEFUL_DURATION_SECONDS} ثانیه"
    )

    print(
        f"مدت کل ضبط: "
        f"{VERIFY_DURATION_SECONDS} ثانیه"
    )

    print(
        f"حداقل Embedding مورد نیاز: "
        f"{MIN_VERIFY_EMBEDDINGS}"
    )

    print(
        f"Threshold: "
        f"{UNLOCK_THRESHOLD:.4f}"
    )

    print(
        f"Minimum margin: "
        f"{MIN_DISTANCE_MARGIN:.4f}"
    )

    print("=" * 60)

    # =====================================================
    # نمایش مسیرها
    # =====================================================

    print()

    print(
        "مسیر پروژه:"
    )

    print(
        PROJECT_DIR
    )

    print()

    print(
        "مسیر مدل:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "دیتابیس کاربران:"
    )

    print(
        "PostgreSQL"
    )

    print()

    print(
        "مسیر Threshold:"
    )

    print(
        THRESHOLD_FILE
    )

    # =====================================================
    # Load model
    # =====================================================

    print()

    print(
        "در حال بارگذاری مدل..."
    )

    if not MODEL_PATH.exists():

        print()

        print(
            "❌ فایل مدل پیدا نشد:"
        )

        print(
            MODEL_PATH
        )

        return

    try:

        model = get_model().to(
            DEVICE
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        # -------------------------------------------------
        # پشتیبانی از checkpointهای مختلف
        # -------------------------------------------------

        if (
            isinstance(
                checkpoint,
                dict
            )
            and "model_state_dict" in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            isinstance(
                checkpoint,
                dict
            )
            and "state_dict" in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict
        )

        model.eval()

    except Exception as e:

        print()

        print(
            "❌ خطا در بارگذاری مدل:"
        )

        print(e)

        return

    print(
        "مدل با موفقیت بارگذاری شد."
    )

    # =====================================================
    # Initialize / Load database
    # =====================================================
    try:
        init_database()
    except Exception as e:
        print()
        print(
            "❌ خطا در اتصال به PostgreSQL:"
        )
        print(e)
        return


    try:

        templates = load_database()

    except Exception as e:

        print()

        print(
            "❌ خطا در بارگذاری دیتابیس:"
        )

        print(e)

        return

    print()

    print(
        f"تعداد کاربران ثبت‌شده: "
        f"{len(templates)}"
    )

    print()

    print(
        "کاربران:"
    )

    for user_name in templates:

        print(
            f"  - {user_name}"
        )

    # =====================================================
    # Record ECG
    # =====================================================

    print()

    print(
        "لطفاً دست‌ها را روی الکترودها "
        "قرار دهید ..."
    )

    print()

    print(
        "در 10 ثانیه اول سعی کنید "
        "دست‌ها ثابت باشند."
    )

    print()

    print(
        "در حال ضبط ECG ..."
    )

    try:

        # مهم:
        # حذف 10 ثانیه اول در این فایل انجام نمی‌شود.
        #
        # signal_to_embeddings خودش 10 ثانیه
        # ابتدایی را حذف می‌کند.

        raw_ecg = record_ecg(
            VERIFY_DURATION_SECONDS
        )

    except TypeError:

        print()

        print(
            "❌ نسخه فعلی record_ecg() "
            "پارامتر مدت ضبط دریافت نمی‌کند."
        )

        print()

        print(
            "در enroll_user.py باید تابع "
            "record_ecg به شکل زیر باشد:"
        )

        print()

        print(
            "def record_ecg(record_seconds=RECORD_SECONDS):"
        )

        return

    except Exception as e:

        print()

        print(
            "❌ ضبط ECG انجام نشد."
        )

        print(e)

        return

    # =====================================================
    # Check samples
    # =====================================================

    expected_samples = int(
        VERIFY_DURATION_SECONDS
        * SAMPLING_RATE
    )

    print()

    print(
        f"نمونه‌های مورد انتظار: "
        f"{expected_samples}"
    )

    print(
        f"نمونه‌های دریافت‌شده: "
        f"{len(raw_ecg)}"
    )

    if len(raw_ecg) < expected_samples:

        print()

        print(
            "❌ تعداد نمونه‌ها کافی نیست."
        )

        # -------------------------------------------------
        # Machine-readable result for app.py
        # -------------------------------------------------

        print(
            "RESULT:DENIED::0.0000:0.0000",
            flush=True
        )

        return

    # =====================================================
    # Signal processing
    # =====================================================

    print()

    print(
        "در حال پردازش ECG ..."
    )

    try:

        embeddings = signal_to_embeddings(
            raw_ecg,
            model
        )

    except Exception as e:

        print()

        print(
            "❌ خطا در پردازش ECG:"
        )

        print(e)

        # -------------------------------------------------
        # Machine-readable result for app.py
        # -------------------------------------------------

        print(
            "RESULT:DENIED::0.0000:0.0000",
            flush=True
        )

        return

    # =====================================================
    # Quality check
    # =====================================================

    print()

    print(
        f"تعداد Embeddingها: "
        f"{len(embeddings)}"
    )

    if (
        len(embeddings)
        < MIN_VERIFY_EMBEDDINGS
    ):

        print()

        print(
            "❌ کیفیت سیگنال کافی نیست."
        )

        print(
            f"Embeddingهای موجود: "
            f"{len(embeddings)}"
        )

        print(
            f"حداقل مورد نیاز: "
            f"{MIN_VERIFY_EMBEDDINGS}"
        )

        print()

        print(
            "این تست وارد مرحله "
            "تشخیص هویت نمی‌شود."
        )

        # -------------------------------------------------
        # Machine-readable result for app.py
        # -------------------------------------------------

        print(
            "RESULT:DENIED::0.0000:0.0000",
            flush=True
        )

        return

    # =====================================================
    # Convert embeddings
    # =====================================================

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # بررسی NaN / Inf
    # -----------------------------------------------------

    if not np.all(
        np.isfinite(embeddings)
    ):

        print()

        print(
            "❌ Embeddingهای تولیدشده "
            "شامل مقادیر نامعتبر هستند."
        )

        # -------------------------------------------------
        # Machine-readable result for app.py
        # -------------------------------------------------

        print(
            "RESULT:DENIED::0.0000:0.0000",
            flush=True
        )

        return

    # =====================================================
    # Individual analysis
    # =====================================================

    analyze_individual_embeddings(
        embeddings,
        templates
    )

    # =====================================================
    # Query embedding
    # =====================================================

    print()

    print("=" * 80)

    print(
        "CREATING QUERY EMBEDDING"
    )

    print("=" * 80)

    query_embedding = np.mean(
        embeddings,
        axis=0
    )

    try:

        query_embedding = (
            normalize_embedding(
                query_embedding
            )
        )

    except ValueError as e:

        print()

        print(
            "❌ Query Embedding نامعتبر است:"
        )

        print(e)

        # -------------------------------------------------
        # Machine-readable result for app.py
        # -------------------------------------------------

        print(
            "RESULT:DENIED::0.0000:0.0000",
            flush=True
        )

        return

    print(
        "Query Embedding با موفقیت "
        "ساخته و L2-normalize شد."
    )

    # =====================================================
    # Final distances
    # =====================================================

    print()

    print("=" * 80)

    print(
        "FINAL DISTANCE USING "
        "MEAN QUERY EMBEDDING"
    )

    print("=" * 80)

    distances = []

    for (
        user_name,
        template
    ) in templates.items():

        distance = calculate_distance(
            query_embedding,
            template
        )

        distances.append(
            (
                user_name,
                distance
            )
        )

        print(
            f"  {user_name:<12}: "
            f"{distance:.4f}"
        )

    # =====================================================
    # Sort
    # =====================================================

    distances.sort(
        key=lambda item: item[1]
    )

    best_user = distances[0][0]

    best_distance = distances[0][1]

    print()

    print(
        "-" * 80
    )

    print(
        f"نزدیک‌ترین تطابق: "
        f"{best_user}"
    )

    print(
        f"فاصله نفر اول: "
        f"{best_distance:.4f}"
    )

    print(
        f"Threshold: "
        f"{UNLOCK_THRESHOLD:.4f}"
    )

    # =====================================================
    # مقدار پیش‌فرض Margin
    # =====================================================

    # برای اینکه در صورت وجود تنها یک Template
    # متغیر margin همچنان مقدار معتبر داشته باشد.

    margin = 0.0

    # =====================================================
    # Threshold check
    # =====================================================

    if (
        best_distance
        >= UNLOCK_THRESHOLD
    ):

        print()

        print(
            "❌ هویت تأیید نشد."
        )

        print(
            "فاصله از Threshold بیشتر "
            "یا مساوی است."
        )

        print(
            f"Distance = "
            f"{best_distance:.4f}"
        )

        print(
            f"Threshold = "
            f"{UNLOCK_THRESHOLD:.4f}"
        )

        print(
            "=" * 80
        )

        # -------------------------------------------------
        # Machine-readable result for app.py
        # -------------------------------------------------
        # در حالت رد شدن، نام کاربر ارسال نمی‌شود؛
        # چون نزدیک‌ترین Template لزوماً هویت واقعی نیست.

        print(
            f"RESULT:DENIED:"
            f":"
            f"{best_distance:.4f}:"
            f"0.0000",
            flush=True
        )

        return

    # =====================================================
    # Margin check
    # =====================================================

    if len(distances) >= 2:

        second_user = (
            distances[1][0]
        )

        second_distance = (
            distances[1][1]
        )

        margin = (
            second_distance
            - best_distance
        )

        print()

        print(
            f"نفر دوم: "
            f"{second_user}"
        )

        print(
            f"فاصله نفر دوم: "
            f"{second_distance:.4f}"
        )

        print(
            f"Margin: "
            f"{margin:.4f}"
        )

        if (
            margin
            < MIN_DISTANCE_MARGIN
        ):

            print()

            print(
                "⚠️ نتیجه نامطمئن است."
            )

            print(
                "فاصله نفر اول و دوم "
                "خیلی نزدیک است."
            )

            print(
                "قفل باز نمی‌شود."
            )

            print(
                "=" * 80
            )

            # -------------------------------------------------
            # Machine-readable result for app.py
            # -------------------------------------------------

            print(
                f"RESULT:DENIED:"
                f":"
                f"{best_distance:.4f}:"
                f"{margin:.4f}",
                flush=True
            )

            return

    # =====================================================
    # Final acceptance
    # =====================================================

    print()

    print("=" * 80)

    print(
        f"✅ هویت تأیید شد: "
        f"{best_user}"
    )

    print(
        "🔓 دسترسی مجاز است."
    )

    print("=" * 80)

    # =====================================================
    # Machine-readable result for app.py
    # =====================================================

    # این خط برای Frontend / Flask است.
    #
    # فرمت:
    #
    # RESULT:SUCCESS:نام کاربر:فاصله:Margin
    #
    # مثال:
    #
    # RESULT:SUCCESS:saratest:0.4195:0.2637

    print(
        f"RESULT:SUCCESS:"
        f"{best_user}:"
        f"{best_distance:.4f}:"
        f"{margin:.4f}",
        flush=True
    )

    # =====================================================
    # Relay
    # =====================================================

    # فعلاً هیچ فرمانی برای رله ارسال نمی‌شود.
    #
    # بعداً می‌توان این بخش را اضافه کرد:
    #
    # send_unlock_command()


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()
