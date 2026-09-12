"""
enroll_user.py

ثبت کاربر جدید برای سامانه احراز هویت ECG.

Pipeline:

MAX30003 CSV / Arduino Serial

    ↓

Discard first 10 seconds

    ↓

Resample 128 Hz → 500 Hz

    ↓

biometric_filter()

    ↓

Z-score normalization

    ↓

QRS detection + R-peak refinement

    ↓

Beat extraction

    ↓

Dual-Beat

    ↓

112×112 grayscale image

    ↓

Siamese CNN embedding

    ↓

Mean embedding = user template

Live ECG:

Arduino

    ↓

record_ecg()

    ↓

results/live_ecg.json

    ↓

Flask app

    ↓

Web UI

"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import serial
import torch
import matplotlib.pyplot as plt
from scipy.signal import resample

from preprocessing import (
    biometric_filter,
    detect_and_refine_qrs,
    extract_beats,
)

from model import get_model


# =========================================================
# Settings
# =========================================================

PORT = "COM5"
BAUDRATE = 115200

ACTUAL_FS = 128
TARGET_FS = 500

RECORD_SECONDS = 35

DISCARD_SECONDS = 10

EXPECTED_SAMPLES = (
    ACTUAL_FS * RECORD_SECONDS
)

DISCARD_SAMPLES = (
    ACTUAL_FS * DISCARD_SECONDS
)

USEFUL_SECONDS = (
    RECORD_SECONDS - DISCARD_SECONDS
)

IMAGE_SIZE = 112
IMAGE_DPI = 100

MIN_ENROLLMENT_EMBEDDINGS = 15

# حداقل تعداد R-peaks قابل قبول
#
# برای 25 ثانیه سیگنال:
# 40 BPM -> حدود 17 ضربان
# 45 BPM -> حدود 19 ضربان
#
# بنابراین 15 را به‌عنوان حداقل مطلق قرار می‌دهیم.

MIN_R_PEAKS = 15

# حداقل تعداد Beat معتبر

MIN_BEATS = 14


# =========================================================
# Live ECG Settings
# =========================================================

# آخرین 512 نمونه = حدود 4 ثانیه در 128Hz

LIVE_ECG_BUFFER_SIZE = 512

# هر چند نمونه یک بار فایل Live ECG به‌روزرسانی شود.
# 8 یعنی تقریباً هر 62.5 میلی‌ثانیه

LIVE_ECG_UPDATE_EVERY = 8


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = (
    BASE_DIR
    / "data"
    / "max30003"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

# ساختار:
#
# results/
# ├── models/
# ├── user_database/
# │   ├── users.csv
# │   ├── sara.npy
# │   └── ...
#
# └── live_ecg.json

USERS_DIR = (
    RESULTS_DIR
    / "user_database"
)

USERS_FILE = (
    USERS_DIR
    / "users.csv"
)

IMAGE_SCALE_FILE = (
    RESULTS_DIR
    / "image_scale.txt"
)

# فایل موقت/اشتراکی برای نمایش زنده ECG

LIVE_ECG_FILE = (
    RESULTS_DIR
    / "live_ecg.json"
)


MODEL_PATH = (
    RESULTS_DIR
    / "models"
    / "finetuned_model.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Utility
# =========================================================

def print_separator():

    print(
        "\n" + "=" * 70
    )


# =========================================================
# Live ECG Writer
# =========================================================

def clear_live_ecg():
    """
    پاک کردن داده‌های Live ECG قبلی.

    در شروع هر Recording اجرا می‌شود.

    برای جلوگیری از خطای موقتی Windows هنگام
    دسترسی هم‌زمان Flask به فایل، عملیات replace
    چند بار تلاش می‌شود.
    """

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "samples": [],
            "sampling_rate": ACTUAL_FS,
            "timestamp": time.time(),
            "recording": False
        }

        temp_file = (
            LIVE_ECG_FILE.with_suffix(".tmp")
        )

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False
            )

        # -------------------------------------------------
        # جایگزینی اتمیک با چند تلاش
        # -------------------------------------------------

        replaced = False

        for attempt in range(5):

            try:

                os.replace(
                    temp_file,
                    LIVE_ECG_FILE
                )

                replaced = True
                break

            except PermissionError:

                if attempt < 4:
                    time.sleep(0.05)
                else:
                    raise

        if not replaced:

            raise PermissionError(
                "نتوانستیم فایل Live ECG را جایگزین کنیم."
            )

    except Exception as e:

        print(
            f"WARNING: خطا در پاک کردن Live ECG: {e}"
        )


def write_live_ecg(
    samples,
    recording=True
):
    """
    ذخیره آخرین نمونه‌های ECG در live_ecg.json.

    برای جلوگیری از حجم زیاد فایل، فقط آخرین
    LIVE_ECG_BUFFER_SIZE نمونه ذخیره می‌شوند.

    نوشتن به صورت atomic انجام می‌شود تا Flask
    فایل ناقص دریافت نکند.

    در صورت دسترسی هم‌زمان Flask به فایل، عملیات
    جایگزینی چند بار تکرار می‌شود.
    """

    try:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        samples = np.asarray(
            samples,
            dtype=np.float64
        )

        samples = samples[
            np.isfinite(samples)
        ]

        if len(samples) > LIVE_ECG_BUFFER_SIZE:

            samples = samples[
                -LIVE_ECG_BUFFER_SIZE:
            ]

        data = {
            "samples": samples.tolist(),
            "sampling_rate": ACTUAL_FS,
            "timestamp": time.time(),
            "recording": recording
        }

        temp_file = (
            LIVE_ECG_FILE.with_suffix(".tmp")
        )

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False
            )

        # -------------------------------------------------
        # جایگزینی اتمیک فایل
        #
        # اگر Flask هم‌زمان فایل را باز کرده باشد،
        # Windows ممکن است موقتاً PermissionError بدهد.
        # -------------------------------------------------

        replaced = False

        for attempt in range(5):

            try:

                os.replace(
                    temp_file,
                    LIVE_ECG_FILE
                )

                replaced = True
                break

            except PermissionError:

                if attempt < 4:
                    time.sleep(0.05)
                else:
                    raise

        if not replaced:

            raise PermissionError(
                "نتوانستیم فایل Live ECG را جایگزین کنیم."
            )

    except Exception as e:

        # خطای Live ECG نباید باعث متوقف شدن
        # فرآیند اصلی Enrollment شود.

        print(
            f"WARNING: خطا در نوشتن Live ECG: {e}"
        )


def finish_live_ecg(samples):
    """
    آخرین وضعیت ECG را ذخیره می‌کند و
    recording را False قرار می‌دهد.
    """

    write_live_ecg(
        samples,
        recording=False
    )


# =========================================================
# Load image scale
# =========================================================

def load_image_scale():
    """
    مقدار suggested_limit را از image_scale.txt می‌خواند.
    """

    if not IMAGE_SCALE_FILE.exists():

        print(
            f"WARNING: فایل image_scale.txt پیدا نشد:\n"
            f"{IMAGE_SCALE_FILE}"
        )

        # مقدار پیش‌فرض

        return 1.0

    suggested_limit = None

    with open(
        IMAGE_SCALE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if "=" in line:

                key, value = line.split(
                    "=",
                    1
                )

                key = key.strip()
                value = value.strip()

                if key == "suggested_limit":

                    try:

                        suggested_limit = float(
                            value
                        )

                    except ValueError:
                        pass

    if (
        suggested_limit is None
        or suggested_limit <= 0
    ):

        print(
            "WARNING: suggested_limit معتبر پیدا نشد. "
            "مقدار پیش‌فرض 1.0 استفاده می‌شود."
        )

        return 1.0

    return suggested_limit


# =========================================================
# Dual Beat → Image Tensor
# =========================================================

def dual_beat_to_tensor(dual_beat):
    """
    تبدیل Dual-Beat شامل 600 نمونه به تصویر 112×112
    و سپس Tensor مناسب برای شبکه.
    """

    if len(dual_beat) != 600:

        raise ValueError(
            f"Dual-beat باید 600 نمونه داشته باشد، "
            f"اما {len(dual_beat)} نمونه دریافت شد."
        )

    suggested_limit = load_image_scale()

    fig = plt.figure(
        figsize=(
            IMAGE_SIZE / IMAGE_DPI,
            IMAGE_SIZE / IMAGE_DPI
        ),
        dpi=IMAGE_DPI,
        frameon=False,
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    x = np.arange(
        len(dual_beat)
    )

    ax.plot(
        x,
        dual_beat,
        color="black",
        linewidth=1.0,
    )

    ax.set_xlim(
        0,
        599
    )

    ax.set_ylim(
        -suggested_limit,
        suggested_limit
    )

    ax.axis("off")

    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )

    fig.canvas.draw()

    # گرفتن تصویر از canvas

    image = np.asarray(
        fig.canvas.buffer_rgba()
    )

    # RGBA → RGB

    image = image[:, :, :3]

    # RGB → grayscale

    image = np.mean(
        image,
        axis=2
    ).astype(
        np.uint8
    )

    plt.close(fig)

    # اطمینان از اندازه

    if image.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE
    ):

        raise RuntimeError(
            f"اندازه تصویر اشتباه است: {image.shape}"
        )

    # تبدیل 0..255 → 0..1

    image = (
        image.astype(
            np.float32
        ) / 255.0
    )

    # [H,W] → [1,H,W]

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor


# =========================================================
# Read ECG from Arduino
# =========================================================

def record_ecg(
    record_seconds=RECORD_SECONDS
):
    """
    دریافت ECG از Arduino Uno.

    پارامتر:
        record_seconds:
            مدت زمان ثبت ECG بر حسب ثانیه.

    در Enrollment به صورت پیش‌فرض 35 ثانیه ثبت می‌شود.

    در Verification نیز verify_unlock.py می‌تواند
    مدت زمان دلخواه را مستقیماً ارسال کند.

    Arduino باید فقط مقدار ECG را در Serial چاپ کند.

    علاوه بر نگهداری نمونه‌ها برای پردازش اصلی،
    آخرین نمونه‌های ECG در live_ecg.json نوشته می‌شوند
    تا Flask بتواند آن‌ها را برای نمایش زنده بخواند.
    """

    # تعداد نمونه مورد انتظار بر اساس مدت زمان واقعی

    expected_samples = int(
        ACTUAL_FS * record_seconds
    )

    print_separator()

    print(
        "شروع دریافت ECG ..."
    )

    print(
        f"Port: {PORT}"
    )

    print(
        f"Baudrate: {BAUDRATE}"
    )

    print(
        f"Record duration: {record_seconds} seconds"
    )

    print(
        f"Expected samples: {expected_samples}"
    )

    # پاک کردن داده قبلی Live ECG

    clear_live_ecg()

    try:

        ser = serial.Serial(
            PORT,
            BAUDRATE,
            timeout=2
        )

    except Exception as e:

        print(
            "\nERROR در باز کردن پورت سریال:"
        )

        print(e)

        # اعلام اینکه Recording شروع نشده

        clear_live_ecg()

        return None

    # صبر برای پایدار شدن Serial

    time.sleep(2)

    # پاک کردن داده‌های قبلی

    ser.reset_input_buffer()

    samples = []

    start_time = time.time()

    try:

        while len(samples) < expected_samples:

            line = ser.readline()

            if not line:
                continue

            try:

                text = line.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if not text:
                    continue

                value = float(text)

                if np.isfinite(value):

                    samples.append(
                        value
                    )

                    # -----------------------------------------
                    # Live ECG
                    # -----------------------------------------
                    #
                    # هر چند نمونه یک بار آخرین 512 نمونه
                    # برای Flask ذخیره می‌شود.
                    #

                    if (
                        len(samples)
                        % LIVE_ECG_UPDATE_EVERY
                        == 0
                    ):

                        write_live_ecg(
                            samples,
                            recording=True
                        )

            except ValueError:

                # خطوط غیرعددی Arduino نادیده گرفته می‌شوند

                continue

            elapsed = (
                time.time()
                - start_time
            )

            # کمی زمان اضافه برای تأخیر Serial

            if (
                elapsed
                > record_seconds + 10
            ):

                print(
                    "\nWARNING: دریافت داده بیشتر از "
                    "زمان مورد انتظار طول کشید."
                )

                break

    finally:

        try:
            ser.close()
        except Exception:
            pass

    print(
        f"\nSamples received: {len(samples)}"
    )

    # آخرین وضعیت Live ECG

    finish_live_ecg(
        samples
    )

    if len(samples) < expected_samples:

        print(
            f"ERROR: تعداد نمونه کافی نیست.\n"
            f"Expected: {expected_samples}\n"
            f"Received: {len(samples)}"
        )

        return None

    samples = np.asarray(
        samples[:expected_samples],
        dtype=np.float64
    )

    return samples


# =========================================================
# Signal → Embeddings
# =========================================================

def signal_to_embeddings(
    raw_ecg,
    model,
    verbose=True
):
    """
    تبدیل ECG خام به مجموعه embedding.

    خروجی:
        list[np.ndarray]
    """

    raw_ecg = np.asarray(
        raw_ecg,
        dtype=np.float64
    )

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if len(raw_ecg) == 0:

        print(
            "ERROR: سیگنال خالی است."
        )

        return []

    if not np.all(
        np.isfinite(raw_ecg)
    ):

        print(
            "ERROR: سیگنال شامل NaN یا Inf است."
        )

        return []

    # -----------------------------------------------------
    # Discard first 10 seconds
    # -----------------------------------------------------

    if len(raw_ecg) <= DISCARD_SAMPLES:

        print(
            "ERROR: طول سیگنال برای حذف 10 ثانیه اول کافی نیست."
        )

        return []

    ecg = raw_ecg[
        DISCARD_SAMPLES:
    ]

    if verbose:

        print(
            f"نمونه‌های اولیه: {len(raw_ecg)}"
        )

        print(
            f"حذف {DISCARD_SECONDS} ثانیه اول "
            f"({DISCARD_SAMPLES} نمونه)"
        )

        print(
            f"نمونه‌های مورد استفاده: {len(ecg)}"
        )

    # -----------------------------------------------------
    # Resample 128 Hz → 500 Hz
    # -----------------------------------------------------

    n_target = int(
        len(ecg)
        * TARGET_FS
        / ACTUAL_FS
    )

    resampled = resample(
        ecg,
        n_target
    )

    fs = TARGET_FS

    if verbose:

        print(
            f"Resample: {len(ecg)} → {len(resampled)}"
        )

        print(
            f"Sampling rate: {fs} Hz"
        )

    # -----------------------------------------------------
    # Biometric filtering
    # -----------------------------------------------------

    try:

        filtered = biometric_filter(
            resampled,
            fs
        )

    except Exception as e:

        print(
            "\nERROR در biometric_filter:"
        )

        print(e)

        return []

    # -----------------------------------------------------
    # Z-score normalization
    # -----------------------------------------------------

    mean_value = np.mean(
        filtered
    )

    std_value = np.std(
        filtered
    )

    if (
        not np.isfinite(std_value)
        or std_value == 0
    ):

        print(
            "ERROR: انحراف معیار سیگنال صفر یا نامعتبر است."
        )

        return []

    biometric_ecg = (
        filtered - mean_value
    ) / std_value

    # -----------------------------------------------------
    # QRS detection + refinement
    # -----------------------------------------------------

    try:

        anchors = detect_and_refine_qrs(
            resampled,
            fs,
            biometric_ecg
        )

    except Exception as e:

        print(
            "\nERROR در تشخیص R-peak:"
        )

        print(e)

        return []

    anchors = np.asarray(
        anchors,
        dtype=int
    )

    # حذف anchorهای خارج از محدوده

    anchors = anchors[
        (anchors >= 0)
        &
        (
            anchors
            < len(biometric_ecg)
        )
    ]

    # حذف موارد تکراری

    anchors = np.unique(
        anchors
    )

    if verbose:

        print(
            f"تعداد R-peaks پیدا شده: "
            f"{len(anchors)}"
        )

    # -----------------------------------------------------
    # Quality check: R-peaks
    # -----------------------------------------------------

    if len(anchors) < MIN_R_PEAKS:

        print(
            "\n" + "!" * 70
        )

        print(
            "ثبت ECG از نظر تعداد ضربان قابل قبول نیست."
        )

        print(
            f"R-peaks پیدا شده: {len(anchors)}"
        )

        print(
            f"حداقل مورد نیاز: {MIN_R_PEAKS}"
        )

        print(
            "\nاین مورد معمولاً می‌تواند ناشی از یکی از "
            "موارد زیر باشد:"
        )

        print(
            "1. اتصال نامناسب الکترودها"
        )

        print(
            "2. نویز زیاد سیگنال"
        )

        print(
            "3. حرکت دست یا بدن"
        )

        print(
            "4. دامنه یا شکل نامناسب سیگنال ECG"
        )

        print(
            "5. از دست رفتن بعضی از QRSها توسط الگوریتم تشخیص"
        )

        print(
            "!" * 70
        )

        return []

    # -----------------------------------------------------
    # Extract beats
    # -----------------------------------------------------

    try:

        beats = extract_beats(
            biometric_ecg,
            anchors,
            fs
        )

    except Exception as e:

        print(
            "\nERROR در استخراج Beatها:"
        )

        print(e)

        return []

    beats = np.asarray(
        beats,
        dtype=np.float64
    )

    if verbose:

        print(
            f"تعداد Beatهای معتبر: {len(beats)}"
        )

    if len(beats) < MIN_BEATS:

        print(
            "\n" + "!" * 70
        )

        print(
            "تعداد Beatهای معتبر برای Enrollment کافی نیست."
        )

        print(
            f"Beats: {len(beats)}"
        )

        print(
            f"Minimum required: {MIN_BEATS}"
        )

        print(
            "!" * 70
        )

        return []

    # -----------------------------------------------------
    # Create Dual-Beats
    # -----------------------------------------------------

    dual_beats = []

    for i in range(
        len(beats) - 1
    ):

        dual_beat = np.concatenate(
            [
                beats[i],
                beats[i + 1]
            ]
        )

        if len(dual_beat) != 600:
            continue

        dual_beats.append(
            dual_beat
        )

    if verbose:

        print(
            f"تعداد Dual-Beatهای معتبر: "
            f"{len(dual_beats)}"
        )

    if (
        len(dual_beats)
        < MIN_ENROLLMENT_EMBEDDINGS
    ):

        print(
            "\n" + "!" * 70
        )

        print(
            "تعداد Embedding قابل تولید "
            "برای Enrollment کافی نیست."
        )

        print(
            f"Dual-Beats: {len(dual_beats)}"
        )

        print(
            f"Minimum required: "
            f"{MIN_ENROLLMENT_EMBEDDINGS}"
        )

        print(
            "!" * 70
        )

        return []

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    embeddings = []

    model.eval()

    with torch.no_grad():

        for dual_beat in dual_beats:

            try:

                img_tensor = dual_beat_to_tensor(
                    dual_beat
                )

                img_tensor = (
                    img_tensor
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                embedding = model.embedding_net(
                    img_tensor
                )

                embedding = (
                    embedding
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                if not np.all(
                    np.isfinite(
                        embedding
                    )
                ):
                    continue

                embeddings.append(
                    embedding
                )

            except Exception as e:

                print(
                    f"WARNING: خطا در ساخت embedding: {e}"
                )

                continue

    if verbose:

        print(
            f"تعداد Embedding ساخته‌شده: "
            f"{len(embeddings)}"
        )

    if (
        len(embeddings)
        < MIN_ENROLLMENT_EMBEDDINGS
    ):

        print(
            "\n" + "!" * 70
        )

        print(
            "Enrollment رد شد."
        )

        print(
            f"Embeddingهای معتبر: {len(embeddings)}"
        )

        print(
            f"حداقل مورد نیاز: "
            f"{MIN_ENROLLMENT_EMBEDDINGS}"
        )

        print(
            "!" * 70
        )

        return []

    return embeddings


# =========================================================
# Save user template
# =========================================================

def save_user_template(
    name,
    embeddings
):
    """
    ساخت template با میانگین embeddingها
    و ذخیره در:

    results/user_database/<name>.npy
    """

    if (
        len(embeddings)
        < MIN_ENROLLMENT_EMBEDDINGS
    ):

        print(
            "ERROR: تعداد embedding برای "
            "ذخیره template کافی نیست."
        )

        return False

    embeddings_array = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # میانگین embeddingها

    template = np.mean(
        embeddings_array,
        axis=0
    )

    # نرمال‌سازی template

    norm = np.linalg.norm(
        template
    )

    if (
        not np.isfinite(norm)
        or norm == 0
    ):

        print(
            "ERROR: template نامعتبر است."
        )

        return False

    template = (
        template / norm
    )

    # -----------------------------------------------------
    # Create user database directory
    # -----------------------------------------------------

    USERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    template_path = (
        USERS_DIR
        / f"{name}.npy"
    )

    np.save(
        template_path,
        template
    )

    print(
        "\nTemplate ذخیره شد:"
    )

    print(
        template_path
    )

    return True


# =========================================================
# Update users.csv
# =========================================================

def update_users_csv(
    name,
    num_beats
):
    """
    ثبت کاربر در users.csv.

    ساختار فایل:

    user_name,num_beats,embedding_path
    """

    USERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    if USERS_FILE.exists():

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                if row:

                    rows.append(
                        row
                    )

    # مسیر template

    template_path = (
        USERS_DIR
        / f"{name}.npy"
    )

    # بررسی اینکه کاربر قبلاً وجود داشته یا نه

    existing_index = None

    for i, row in enumerate(
        rows
    ):

        existing_name = (
            row.get("user_name")
            or row.get("name")
            or ""
        ).strip().lower()

        if existing_name == name.lower():

            existing_index = i

            break

    new_row = {

        "user_name": name,

        "num_beats": str(
            num_beats
        ),

        "embedding_path": str(
            template_path
        )
    }

    if existing_index is None:

        rows.append(
            new_row
        )

    else:

        rows[
            existing_index
        ] = new_row

    # ذخیره مجدد CSV

    with open(
        USERS_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        fieldnames = [
            "user_name",
            "num_beats",
            "embedding_path"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        "\nusers.csv به‌روزرسانی شد:"
    )

    print(
        USERS_FILE
    )


# =========================================================
# Load Model
# =========================================================

def load_embedding_model():
    """
    بارگذاری مدل Fine-tuned.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"مدل پیدا نشد:\n{MODEL_PATH}"
        )

    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    model = get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # پشتیبانی از چند حالت ذخیره checkpoint

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Model loaded on: {DEVICE}"
    )

    return model


# =========================================================
# Main Enrollment
# =========================================================

def main(name=None):
    if name is None:
        name = input(
            "نام کاربر جدید را وارد کنید: "
        ).strip()
    else:
        name = str(name).strip()

    print_separator()

    print(
        "ECG BIOMETRIC ENROLLMENT"
    )

    print_separator()

    # -----------------------------------------------------
    # User name
    # -----------------------------------------------------



    if not name:

        print(
            "ERROR: نام کاربر نمی‌تواند خالی باشد."
        )

        return

    # جلوگیری از مشکلات نام فایل

    invalid_chars = (
        '<>:"/\\|?*'
    )

    if any(
        char in name
        for char in invalid_chars
    ):

        print(
            "ERROR: نام شامل کاراکتر غیرمجاز است."
        )

        return

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    try:

        model = load_embedding_model()

    except Exception as e:

        print(
            "\nERROR در بارگذاری مدل:"
        )

        print(e)

        return

    # -----------------------------------------------------
    # Record ECG
    # -----------------------------------------------------

    print_separator()

    print(
        "لطفاً الکترودها را به‌درستی متصل کنید."
    )

    print(
        "در طول ثبت تا حد امکان ثابت بمانید."
    )

    print(
        "\nشروع ثبت ECG ..."
    )

    # در Enrollment مدت پیش‌فرض 35 ثانیه است

    raw_ecg = record_ecg()

    if raw_ecg is None:
        return

    # -----------------------------------------------------
    # Process signal
    # -----------------------------------------------------

    print_separator()

    print(
        "در حال پردازش ECG ..."
    )

    embeddings = signal_to_embeddings(
        raw_ecg,
        model
    )

    # -----------------------------------------------------
    # Enrollment failed
    # -----------------------------------------------------

    if (
        len(embeddings)
        < MIN_ENROLLMENT_EMBEDDINGS
    ):

        print_separator()

        print(
            "❌ Enrollment انجام نشد."
        )

        print(
            "هیچ Template جدیدی ذخیره نشد."
        )

        return

    # -----------------------------------------------------
    # Save template
    # -----------------------------------------------------

    print_separator()

    print(
        "تعداد embeddingهای نهایی:"
    )

    print(
        len(embeddings)
    )

    success = save_user_template(
        name,
        embeddings
    )

    if not success:
        return

    # -----------------------------------------------------
    # Update users.csv
    # -----------------------------------------------------

    update_users_csv(
        name,
        len(embeddings)
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print_separator()

    print(
        f"✅ کاربر «{name}» با موفقیت ثبت شد."
    )

    print(
        f"Embedding count: {len(embeddings)}"
    )

    print(
        f"Template: results/user_database/{name}.npy"
    )

    print(
        f"Users database: {USERS_FILE}"
    )

    print_separator()


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()