from flask import Flask, render_template, request, jsonify
from pathlib import Path
import subprocess
import sys
import threading
import time
import re
import json
import os

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


# =========================================================
# Flask
# =========================================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)


# =========================================================
# Paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
VERIFY_SCRIPT = SRC_DIR / "verify_unlock.py"
ENROLL_SCRIPT = SRC_DIR / "enroll_user.py"
RESULTS_DIR = PROJECT_DIR / "results"
USERS_DIR = RESULTS_DIR / "user_database"
USERS_FILE = USERS_DIR / "users.csv"

# ---------------------------------------------------------
# Live ECG file
# ---------------------------------------------------------

LIVE_ECG_FILE = RESULTS_DIR / "live_ecg.json"


# =========================================================
# Arduino
# =========================================================

PORT = "COM5"
BAUDRATE = 115200


# =========================================================
# Global process state
# =========================================================

state_lock = threading.Lock()

system_state = {
    "mode": "idle",
    "status": "idle",
    "message": "سیستم آماده است.",

    # در verification این مقدار توسط مدل تعیین می‌شود
    "username": "",

    "success": False,
    "error_type": "",
    "output": "",
    "distance": None,
    "margin": None,
    "started_at": None,
    "finished_at": None
}

current_process = None


# =========================================================
# Utility
# =========================================================

def set_state(**kwargs):
    global system_state

    with state_lock:
        system_state.update(kwargs)


def get_state():
    with state_lock:
        return dict(system_state)


# =========================================================
# Live ECG
# =========================================================

def get_live_ecg():
    """
    خواندن آخرین داده‌های ECG از live_ecg.json.

    enroll_user.py / verify_unlock.py
    مسئول دریافت داده از Arduino هستند.

    Flask فقط فایل را می‌خواند و هرگز COM5 را
    برای Live ECG باز نمی‌کند.
    """

    default_data = {
        "samples": [],
        "sampling_rate": 128,
        "timestamp": None,
        "recording": False
    }

    if not LIVE_ECG_FILE.exists():
        return default_data

    try:
        with open(
            LIVE_ECG_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return default_data

        samples = data.get(
            "samples",
            []
        )

        if not isinstance(
            samples,
            list
        ):
            samples = []

        # فقط مقادیر عددی معتبر
        clean_samples = []

        for value in samples:
            try:
                value = float(value)

                if (
                    value == value
                    and abs(value) != float("inf")
                ):
                    clean_samples.append(
                        value
                    )

            except (
                TypeError,
                ValueError
            ):
                continue

        return {
            "samples": clean_samples,

            "sampling_rate": data.get(
                "sampling_rate",
                128
            ),

            "timestamp": data.get(
                "timestamp"
            ),

            "recording": bool(
                data.get(
                    "recording",
                    False
                )
            )
        }

    except (
        json.JSONDecodeError,
        OSError,
        ValueError,
        TypeError
    ):
        # ممکن است Flask دقیقاً در لحظه‌ای
        # فایل را بخواند که در حال جایگزینی است.
        # در این حالت داده قبلی/خالی برگردانده می‌شود.
        return default_data


# =========================================================
# Check Arduino COM Port
# =========================================================

def check_arduino_connection():

    if serial is None or list_ports is None:
        return {
            "connected": False,
            "error": "pyserial نصب نیست."
        }

    try:
        ports = list(
            list_ports.comports()
        )

        for port in ports:

            if (
                port.device.upper()
                == PORT.upper()
            ):
                return {
                    "connected": True,
                    "error": ""
                }

        return {
            "connected": False,
            "error": (
                f"پورت {PORT} پیدا نشد."
            )
        }

    except Exception as e:

        return {
            "connected": False,
            "error": str(e)
        }


# =========================================================
# Check User
#
# این تابع فقط برای ENROLLMENT استفاده می‌شود.
# در verification دیگر username لازم نیست.
# =========================================================

def user_exists(username):

    if not USERS_FILE.exists():
        return False

    try:

        import csv

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                name = row.get(
                    "user_name",
                    ""
                ).strip()

                if (
                    name.lower()
                    == username.lower()
                ):
                    return True

    except Exception:
        return False

    return False


# =========================================================
# Start Verification
# =========================================================

@app.route(
    "/api/verify",
    methods=["POST"]
)
def start_verify():

    global current_process

    # -----------------------------------------------------
    # جلوگیری از اجرای همزمان
    # -----------------------------------------------------

    with state_lock:

        if system_state["status"] in [
            "checking",
            "recording",
            "processing",
            "analysis",
            "enrolling"
        ]:

            return jsonify({
                "success": False,
                "message": (
                    "یک فرآیند دیگر در حال اجراست."
                )
            }), 409

    # -----------------------------------------------------
    # بررسی Arduino
    # -----------------------------------------------------

    connection = check_arduino_connection()

    if not connection["connected"]:

        set_state(
            mode="verify",
            status="connection_error",

            message=(
                "اتصال به سنسور برقرار نیست. "
                "لطفاً Arduino و اتصال MAX30003 را بررسی کنید."
            ),

            username="",
            success=False,
            error_type="arduino_not_found",
            output="",
            distance=None,
            margin=None
        )

        return jsonify({
            "success": False,
            "message": get_state()["message"]
        })

    # -----------------------------------------------------
    # شروع verification
    #
    # دیگر username از frontend دریافت نمی‌شود.
    # -----------------------------------------------------

    set_state(
        mode="verify",
        status="checking",

        message=(
            "اتصال به سنسور برقرار شد."
        ),

        username="",
        success=False,
        error_type="",
        output="",
        distance=None,
        margin=None,
        started_at=time.time(),
        finished_at=None
    )

    thread = threading.Thread(
        target=run_verification,
        daemon=True
    )

    thread.start()

    return jsonify({
        "success": True,
        "message": "احراز هویت آغاز شد."
    })


# =========================================================
# Run Verification Script
# =========================================================

def run_verification():

    global current_process

    output_lines = []
    result_received = False

    try:

        set_state(
            status="recording",
            message=(
                "در حال ضبط سیگنال ECG..."
            )
        )

        # -------------------------------------------------
        # اجرای verify_unlock.py
        #
        # هیچ username به آن ارسال نمی‌شود.
        #
        # PYTHONIOENCODING=utf-8
        # -------------------------------------------------

        process_env = dict(
            os.environ
        )

        process_env[
            "PYTHONIOENCODING"
        ] = "utf-8"

        process_env[
            "PYTHONUTF8"
        ] = "1"

        current_process = subprocess.Popen(
            [
                sys.executable,
                str(VERIFY_SCRIPT)
            ],

            cwd=str(PROJECT_DIR),

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,

            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=process_env
        )

        # -------------------------------------------------
        # خواندن خروجی
        # -------------------------------------------------

        for line in current_process.stdout:

            line = line.strip()

            if not line:
                continue

            output_lines.append(
                line
            )

            set_state(
                output="\n".join(
                    output_lines
                )
            )

            # ---------------------------------------------
            # RESULT:SUCCESS
            #
            # فرمت:
            #
            # RESULT:SUCCESS:saratest:0.4545:0.2109
            # ---------------------------------------------

            if line.startswith(
                "RESULT:SUCCESS:"
            ):

                parts = line.split(":")

                if len(parts) >= 5:

                    username = (
                        parts[2].strip()
                    )

                    try:
                        distance = float(
                            parts[3]
                        )
                    except ValueError:
                        distance = None

                    try:
                        margin = float(
                            parts[4]
                        )
                    except ValueError:
                        margin = None

                    # -------------------------------------
                    # username باید واقعاً وجود داشته باشد
                    # -------------------------------------

                    if username:

                        result_received = True

                        set_state(
                            status="success",

                            message=(
                                f"کاربر «{username}» "
                                "تأیید شد — قفل باز شد."
                            ),

                            username=username,
                            success=True,
                            distance=distance,
                            margin=margin,
                            error_type="",
                            finished_at=time.time()
                        )

            # ---------------------------------------------
            # RESULT:DENIED
            #
            # فرمت:
            #
            # RESULT:DENIED::0.98:0.02
            #
            # یا:
            #
            # RESULT:DENIED:username:0.98:0.02
            # ---------------------------------------------

            elif line.startswith(
                "RESULT:DENIED:"
            ):

                parts = line.split(":")

                username = ""
                distance = None
                margin = None

                if len(parts) >= 5:

                    username = (
                        parts[2].strip()
                    )

                    try:
                        distance = float(
                            parts[3]
                        )
                    except ValueError:
                        distance = None

                    try:
                        margin = float(
                            parts[4]
                        )
                    except ValueError:
                        margin = None

                elif len(parts) >= 4:

                    try:
                        distance = float(
                            parts[2]
                        )
                    except ValueError:
                        distance = None

                    try:
                        margin = float(
                            parts[3]
                        )
                    except ValueError:
                        margin = None

                result_received = True

                set_state(
                    status="denied",

                    message=(
                        "هویت کاربر شناسایی نشد — "
                        "دسترسی رد شد."
                    ),

                    username="",
                    success=False,
                    distance=distance,
                    margin=margin,
                    finished_at=time.time()
                )

            # ---------------------------------------------
            # خروجی‌های معمول برنامه
            # ---------------------------------------------

            lower = line.lower()

            if (
                "record" in lower
                or "recording" in lower
                or "ضبط" in line
            ):

                if not result_received:

                    set_state(
                        status="recording",

                        message=(
                            "در حال ضبط سیگنال ECG..."
                        )
                    )

            elif (
                "process" in lower
                or "processing" in lower
                or "پردازش" in line
            ):

                if not result_received:

                    set_state(
                        status="processing",

                        message=(
                            "در حال پردازش سیگنال ECG..."
                        )
                    )

            elif (
                "embedding" in lower
                or "cnn" in lower
                or "تحلیل" in line
                or "ویژگی" in line
            ):

                if not result_received:

                    set_state(
                        status="analysis",

                        message=(
                            "در حال تحلیل بیومتریک..."
                        )
                    )

            # ---------------------------------------------
            # Connection error
            # ---------------------------------------------

            # فقط خطاهایی که واقعاً مربوط به COM5 هستند
            # باید connection_error ایجاد کنند.
            #
            # خطاهایی مثل:
            #
            # [WinError 5] Access is denied:
            # '...live_ecg.tmp' -> '...live_ecg.json'
            #
            # مربوط به فایل Live ECG هستند و نباید
            # به‌عنوان قطع شدن Arduino تشخیص داده شوند.

            serial_error = (

                "serialexception" in lower

                or

                "could not open port 'com5'" in lower

                or

                'could not open port "com5"' in lower

                or

                (
                    "com5" in lower
                    and (
                        "access is denied" in lower
                        or "permission denied" in lower
                        or "busy" in lower
                    )
                )
            )

            if serial_error:

                result_received = True

                set_state(
                    status="connection_error",

                    message=(
                        "اتصال به سنسور برقرار نیست. "
                        "لطفاً Arduino و MAX30003 را بررسی کنید."
                    ),

                    username="",
                    success=False,
                    error_type="serial_error",
                    finished_at=time.time()
                )

        # -------------------------------------------------
        # پایان process
        # -------------------------------------------------

        current_process.wait()

        complete_output = "\n".join(
            output_lines
        )

        lower_output = (
            complete_output.lower()
        )

        current = get_state()

        # =================================================
        # اگر RESULT صریح دریافت شده باشد
        # =================================================

        if result_received:
            return

        # =================================================
        # اگر process با خطا تمام شده باشد
        # =================================================

        if current_process.returncode != 0:

            set_state(
                status="error",

                message=(
                    "در اجرای فرآیند "
                    "احراز هویت خطایی رخ داد."
                ),

                success=False,
                username="",
                error_type="process_error",
                finished_at=time.time()
            )

            return

        # =================================================
        # Fallback
        # =================================================

        denied_keywords = [

            "rejected",
            "access denied",
            "not verified",
            "verification failed",

            "هویت تأیید نشد",
            "هویت تایید نشد",
            "دسترسی رد شد",
            "احراز هویت ناموفق"
        ]

        if any(
            keyword.lower()
            in lower_output
            for keyword in denied_keywords
        ):

            set_state(
                status="denied",

                message=(
                    "هویت تأیید نشد — "
                    "دسترسی رد شد."
                ),

                username="",
                success=False,
                finished_at=time.time()
            )

        else:

            set_state(
                status="finished",

                message=(
                    "فرآیند احراز هویت پایان یافت، "
                    "اما نتیجه معتبر دریافت نشد."
                ),

                username="",
                success=False,
                error_type="result_not_received",
                finished_at=time.time()
            )

    except Exception as e:

        set_state(
            status="error",

            message=(
                f"خطای سیستم: {str(e)}"
            ),

            username="",
            success=False,
            error_type="exception",
            finished_at=time.time()
        )

    finally:

        current_process = None


# =========================================================
# Start Enrollment
# =========================================================

@app.route(
    "/api/enroll",
    methods=["POST"]
)
def start_enrollment():

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "success": False,
            "message": (
                "اطلاعاتی دریافت نشد."
            )
        }), 400

    username = data.get(
        "username",
        ""
    ).strip()

    if not username:

        return jsonify({
            "success": False,
            "message": (
                "لطفاً نام کاربر را وارد کنید."
            )
        }), 400

    # -----------------------------------------------------
    # نام کاربری
    # -----------------------------------------------------

    if not re.match(
        r"^[a-zA-Z0-9_\-]+$",
        username
    ):

        return jsonify({
            "success": False,
            "message": (
                "نام کاربری فقط باید شامل "
                "حروف انگلیسی، عدد، _ یا - باشد."
            )
        }), 400

    # -----------------------------------------------------
    # بررسی تکراری نبودن کاربر
    # -----------------------------------------------------

    if user_exists(username):

        return jsonify({
            "success": False,
            "message": (
                "این کاربر قبلاً ثبت شده است."
            )
        }), 409

    # -----------------------------------------------------
    # جلوگیری از اجرای همزمان
    # -----------------------------------------------------

    with state_lock:

        if system_state["status"] in [
            "checking",
            "recording",
            "processing",
            "analysis",
            "enrolling"
        ]:

            return jsonify({
                "success": False,
                "message": (
                    "یک فرآیند دیگر "
                    "در حال اجراست."
                )
            }), 409

    # -----------------------------------------------------
    # بررسی Arduino
    # -----------------------------------------------------

    connection = check_arduino_connection()

    if not connection["connected"]:

        set_state(
            mode="enroll",
            status="connection_error",

            message=(
                "اتصال به سنسور برقرار نیست. "
                "لطفاً Arduino و MAX30003 را بررسی کنید."
            ),

            username=username,
            success=False,
            error_type="arduino_not_found"
        )

        return jsonify({
            "success": False,
            "message": get_state()["message"]
        })

    # -----------------------------------------------------
    # شروع enrollment
    # -----------------------------------------------------

    set_state(
        mode="enroll",
        status="enrolling",

        message=(
            "در حال آماده‌سازی ثبت کاربر..."
        ),

        username=username,
        success=False,
        error_type="",
        output="",
        distance=None,
        margin=None,
        started_at=time.time(),
        finished_at=None
    )

    thread = threading.Thread(
        target=run_enrollment,
        args=(username,),
        daemon=True
    )

    thread.start()

    return jsonify({
        "success": True,
        "message": "ثبت کاربر آغاز شد."
    })


# =========================================================
# Run Enrollment
# =========================================================

def run_enrollment(username):

    global current_process

    output_lines = []

    try:

        # -------------------------------------------------
        # محیط UTF-8
        # -------------------------------------------------

        process_env = dict(
            os.environ
        )

        process_env[
            "PYTHONIOENCODING"
        ] = "utf-8"

        process_env[
            "PYTHONUTF8"
        ] = "1"

        current_process = subprocess.Popen(
            [
                sys.executable,
                str(ENROLL_SCRIPT)
            ],

            cwd=str(PROJECT_DIR),

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,

            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=process_env
        )

        # -------------------------------------------------
        # ارسال نام کاربر
        # -------------------------------------------------

        try:

            current_process.stdin.write(
                username + "\n"
            )

            current_process.stdin.flush()

        except Exception:
            pass

        set_state(
            status="recording",

            message=(
                "در حال ثبت سیگنال ECG "
                "برای کاربر جدید..."
            )
        )

        # -------------------------------------------------
        # خواندن خروجی
        # -------------------------------------------------

        for line in current_process.stdout:

            line = line.strip()

            if not line:
                continue

            output_lines.append(
                line
            )

            set_state(
                output="\n".join(
                    output_lines
                )
            )

            lower = line.lower()

            if (
                "record" in lower
                or "recording" in lower
                or "ضبط" in line
            ):

                set_state(
                    status="recording",

                    message=(
                        "در حال ضبط سیگنال ECG..."
                    )
                )

            elif (
                "process" in lower
                or "processing" in lower
                or "پردازش" in line
            ):

                set_state(
                    status="processing",

                    message=(
                        "در حال پردازش سیگنال..."
                    )
                )

            elif (
                "embedding" in lower
                or "ویژگی" in line
            ):

                set_state(
                    status="analysis",

                    message=(
                        "در حال استخراج ویژگی..."
                    )
                )

        current_process.wait()

        complete_output = "\n".join(
            output_lines
        )

        lower_output = (
            complete_output.lower()
        )

        # -------------------------------------------------
        # بررسی فایل template
        # -------------------------------------------------

        template_file = (
            USERS_DIR
            / f"{username}.npy"
        )

        saved = template_file.exists()

        # -------------------------------------------------
        # Success
        # -------------------------------------------------

        success_keywords = [

            "saved",
            "success",
            "successfully",
            "ثبت شد",
            "ذخیره شد",
            "موفق"
        ]

        if saved:

            set_state(
                status="enroll_success",

                message=(
                    f"کاربر «{username}» "
                    "با موفقیت ثبت شد."
                ),

                success=True,
                finished_at=time.time()
            )

        elif (
            any(
                keyword.lower()
                in lower_output
                for keyword in success_keywords
            )
            and
            current_process.returncode == 0
        ):

            set_state(
                status="enroll_success",

                message=(
                    f"ثبت کاربر «{username}» "
                    "با موفقیت انجام شد."
                ),

                success=True,
                finished_at=time.time()
            )

        elif current_process.returncode != 0:

            set_state(
                status="error",

                message=(
                    "ثبت کاربر با خطا مواجه شد."
                ),

                success=False,
                error_type="enrollment_error",
                finished_at=time.time()
            )

        else:

            set_state(
                status="finished",

                message=(
                    "فرآیند ثبت کاربر پایان یافت."
                ),

                success=False,
                finished_at=time.time()
            )

    except Exception as e:

        set_state(
            status="error",

            message=(
                f"خطای سیستم: {str(e)}"
            ),

            success=False,
            error_type="exception",
            finished_at=time.time()
        )

    finally:

        current_process = None


# =========================================================
# Status
# =========================================================

@app.route("/api/status")
def api_status():

    state = get_state()

    # -----------------------------------------------------
    # Live ECG data
    # -----------------------------------------------------

    live_ecg = get_live_ecg()

    # اضافه کردن ECG به response

    state["ecg_samples"] = (
        live_ecg["samples"]
    )

    state["ecg_sampling_rate"] = (
        live_ecg["sampling_rate"]
    )

    state["ecg_timestamp"] = (
        live_ecg["timestamp"]
    )

    state["ecg_recording"] = (
        live_ecg["recording"]
    )

    return jsonify(
        state
    )


# =========================================================
# Reset
# =========================================================

@app.route(
    "/api/reset",
    methods=["POST"]
)
def reset():

    set_state(
        mode="idle",
        status="idle",
        message="سیستم آماده است.",
        username="",
        success=False,
        error_type="",
        output="",
        distance=None,
        margin=None,
        started_at=None,
        finished_at=None
    )

    return jsonify({
        "success": True
    })


# =========================================================
# Home
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 60
    )

    print(
        "             ECG SMART LOCK"
    )

    print(
        "=" * 60
    )

    print(
        f"Project: {PROJECT_DIR}"
    )

    print(
        f"Arduino: {PORT}"
    )

    print()

    print(
        "Open browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "=" * 60
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )