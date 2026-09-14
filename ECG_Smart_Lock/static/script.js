let polling = null;


// =========================================================
// ECG Live Signal
// =========================================================

let ecgCanvas = null;
let ecgCtx = null;

let enrollEcgCanvas = null;
let enrollEcgCtx = null;

const ECG_MAX_SAMPLES = 512;
const ECG_POLL_INTERVAL = 100;


// =========================================================
// Initialize ECG
// =========================================================

function initializeECG() {

    // -----------------------------------------------------
    // Verification ECG
    // -----------------------------------------------------

    ecgCanvas = document.getElementById("ecgCanvas");

    if (ecgCanvas) {

        ecgCtx = ecgCanvas.getContext("2d");

        resizeECGCanvas();
    }
    else {

        console.warn(
            "Verification ECG canvas not found."
        );
    }


    // -----------------------------------------------------
    // Enrollment ECG
    // -----------------------------------------------------

    enrollEcgCanvas =
        document.getElementById("enrollEcgCanvas");

    if (enrollEcgCanvas) {

        enrollEcgCtx =
            enrollEcgCanvas.getContext("2d");

        resizeEnrollmentECGCanvas();
    }


    // -----------------------------------------------------
    // Resize
    // -----------------------------------------------------

    window.addEventListener(
        "resize",
        function () {

            resizeECGCanvas();

            resizeEnrollmentECGCanvas();
        }
    );


    // -----------------------------------------------------
    // Clear
    // -----------------------------------------------------

    clearECGCanvas();

    clearEnrollmentECGCanvas();
}


// =========================================================
// Resize Verification ECG Canvas
// =========================================================

function resizeECGCanvas() {

    if (!ecgCanvas || !ecgCtx) {
        return;
    }

    const rect =
        ecgCanvas.getBoundingClientRect();

    if (
        rect.width <= 0 ||
        rect.height <= 0
    ) {
        return;
    }

    const dpr =
        window.devicePixelRatio || 1;

    ecgCanvas.width =
        Math.max(
            1,
            Math.floor(rect.width * dpr)
        );

    ecgCanvas.height =
        Math.max(
            1,
            Math.floor(rect.height * dpr)
        );

    ecgCtx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );

    clearECGCanvas();
}


// =========================================================
// Resize Enrollment ECG Canvas
// =========================================================

function resizeEnrollmentECGCanvas() {

    if (
        !enrollEcgCanvas ||
        !enrollEcgCtx
    ) {
        return;
    }

    const rect =
        enrollEcgCanvas.getBoundingClientRect();

    if (
        rect.width <= 0 ||
        rect.height <= 0
    ) {
        return;
    }

    const dpr =
        window.devicePixelRatio || 1;

    enrollEcgCanvas.width =
        Math.max(
            1,
            Math.floor(rect.width * dpr)
        );

    enrollEcgCanvas.height =
        Math.max(
            1,
            Math.floor(rect.height * dpr)
        );

    enrollEcgCtx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );

    clearEnrollmentECGCanvas();
}


// =========================================================
// Clear Verification ECG
// =========================================================

function clearECGCanvas() {

    if (
        !ecgCanvas ||
        !ecgCtx
    ) {
        return;
    }

    const width =
        ecgCanvas.clientWidth;

    const height =
        ecgCanvas.clientHeight;

    if (
        width <= 0 ||
        height <= 0
    ) {
        return;
    }

    ecgCtx.clearRect(
        0,
        0,
        width,
        height
    );

    drawECGGrid(
        ecgCtx,
        width,
        height
    );
}


// =========================================================
// Clear Enrollment ECG
// =========================================================

function clearEnrollmentECGCanvas() {

    if (
        !enrollEcgCanvas ||
        !enrollEcgCtx
    ) {
        return;
    }

    const width =
        enrollEcgCanvas.clientWidth;

    const height =
        enrollEcgCanvas.clientHeight;

    if (
        width <= 0 ||
        height <= 0
    ) {
        return;
    }

    enrollEcgCtx.clearRect(
        0,
        0,
        width,
        height
    );

    drawECGGrid(
        enrollEcgCtx,
        width,
        height
    );
}


// =========================================================
// Draw ECG Grid
// =========================================================

function drawECGGrid(
    ctx,
    width,
    height
) {

    if (
        !ctx ||
        width <= 0 ||
        height <= 0
    ) {
        return;
    }


    // -----------------------------------------------------
    // Background
    // -----------------------------------------------------

    ctx.fillStyle = "#f0f9ff";

    ctx.fillRect(
        0,
        0,
        width,
        height
    );


    // -----------------------------------------------------
    // Small Grid
    // -----------------------------------------------------

    ctx.strokeStyle =
        "rgba(59, 130, 246, 0.08)";

    ctx.lineWidth = 1;

    const smallGrid = 40;

    // Vertical lines

    for (
        let x = 0;
        x <= width;
        x += smallGrid
    ) {

        ctx.beginPath();

        ctx.moveTo(
            x,
            0
        );

        ctx.lineTo(
            x,
            height
        );

        ctx.stroke();
    }


    // Horizontal lines

    for (
        let y = 0;
        y <= height;
        y += smallGrid
    ) {

        ctx.beginPath();

        ctx.moveTo(
            0,
            y
        );

        ctx.lineTo(
            width,
            y
        );

        ctx.stroke();
    }


    // -----------------------------------------------------
    // Large Grid
    // -----------------------------------------------------

    ctx.strokeStyle =
        "rgba(59, 130, 246, 0.12)";

    const largeGrid = 160;

    // Vertical lines

    for (
        let x = 0;
        x <= width;
        x += largeGrid
    ) {

        ctx.beginPath();

        ctx.moveTo(
            x,
            0
        );

        ctx.lineTo(
            x,
            height
        );

        ctx.stroke();
    }


    // Horizontal lines

    for (
        let y = 0;
        y <= height;
        y += largeGrid
    ) {

        ctx.beginPath();

        ctx.moveTo(
            0,
            y
        );

        ctx.lineTo(
            width,
            y
        );

        ctx.stroke();
    }
}


// =========================================================
// Normalize ECG Samples
// =========================================================

function prepareECGSamples(samples) {

    if (!Array.isArray(samples)) {
        return [];
    }

    let values =
        samples
            .map(Number)
            .filter(Number.isFinite);


    if (values.length < 2) {
        return [];
    }


    // Keep only latest samples

    if (
        values.length >
        ECG_MAX_SAMPLES
    ) {

        values =
            values.slice(
                -ECG_MAX_SAMPLES
            );
    }

    return values;
}


// =========================================================
// Draw ECG On Canvas
// =========================================================

function drawECGOnCanvas(
    canvas,
    ctx,
    values
) {

    if (
        !canvas ||
        !ctx ||
        values.length < 2
    ) {
        return;
    }


    const width =
        canvas.clientWidth;

    const height =
        canvas.clientHeight;


    if (
        width <= 0 ||
        height <= 0
    ) {
        return;
    }


    // -----------------------------------------------------
    // Background + Grid
    // -----------------------------------------------------

    drawECGGrid(
        ctx,
        width,
        height
    );


    // -----------------------------------------------------
    // Find Signal Range
    // -----------------------------------------------------

    let minValue =
        Math.min(...values);

    let maxValue =
        Math.max(...values);

    let range =
        maxValue - minValue;


    // Prevent division by zero

    if (
        !Number.isFinite(range) ||
        range < 0.000001
    ) {

        range = 1;

        minValue -= 0.5;
        maxValue += 0.5;
    }


    // -----------------------------------------------------
    // Vertical Padding
    // -----------------------------------------------------

    const padding =
        range * 0.12;

    minValue -= padding;
    maxValue += padding;

    range =
        maxValue - minValue;


    // -----------------------------------------------------
    // Draw ECG
    // -----------------------------------------------------

    ctx.beginPath();

    ctx.lineWidth = 2.5;

    ctx.strokeStyle =
        "#06b6d4";

    ctx.lineJoin =
        "round";

    ctx.lineCap =
        "round";

    ctx.shadowColor =
        "rgba(6, 182, 212, 0.5)";

    ctx.shadowBlur =
        12;


    for (
        let i = 0;
        i < values.length;
        i++
    ) {

        const x =
            (
                i /
                (values.length - 1)
            ) * width;


        const normalized =
            (
                values[i] -
                minValue
            ) / range;


        const y =
            height -
            normalized * height;


        if (i === 0) {

            ctx.moveTo(
                x,
                y
            );
        }
        else {

            ctx.lineTo(
                x,
                y
            );
        }
    }


    ctx.stroke();

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;


    // -----------------------------------------------------
    // Current Signal Point
    // -----------------------------------------------------

    const lastIndex =
        values.length - 1;


    const lastX =
        (
            lastIndex /
            (values.length - 1)
        ) * width;


    const lastNormalized =
        (
            values[lastIndex] -
            minValue
        ) / range;


    const lastY =
        height -
        lastNormalized * height;


    ctx.beginPath();

    ctx.fillStyle =
        "#06b6d4";

    ctx.shadowColor =
        "rgba(6, 182, 212, 0.6)";

    ctx.shadowBlur =
        16;

    ctx.arc(
        lastX,
        lastY,
        5,
        0,
        Math.PI * 2
    );

    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
}


// =========================================================
// Draw Live ECG
// =========================================================

function drawECG(samples) {

    const values =
        prepareECGSamples(
            samples
        );


    if (values.length < 2) {
        return;
    }


    // -----------------------------------------------------
    // Verification Page
    // -----------------------------------------------------

    if (
        ecgCanvas &&
        ecgCtx
    ) {

        drawECGOnCanvas(
            ecgCanvas,
            ecgCtx,
            values
        );
    }


    // -----------------------------------------------------
    // Enrollment Page
    // -----------------------------------------------------

    if (
        enrollEcgCanvas &&
        enrollEcgCtx
    ) {

        drawECGOnCanvas(
            enrollEcgCanvas,
            enrollEcgCtx,
            values
        );
    }
}


// =========================================================
// Update ECG From Server
// =========================================================

function updateECGFromStatus(data) {

    if (!data) {
        return;
    }


    if (
        Array.isArray(
            data.ecg_samples
        ) &&
        data.ecg_samples.length >= 2
    ) {

        drawECG(
            data.ecg_samples
        );
    }
}


// =========================================================
// Mode
// =========================================================

function switchMode(mode) {

    const verifyPage =
        document.getElementById(
            "verifyPage"
        );

    const enrollPage =
        document.getElementById(
            "enrollPage"
        );

    const verifyTab =
        document.getElementById(
            "verifyTab"
        );

    const enrollTab =
        document.getElementById(
            "enrollTab"
        );


    if (
        mode === "verify"
    ) {

        if (verifyPage) {
            verifyPage.classList.remove(
                "hidden"
            );
        }

        if (enrollPage) {
            enrollPage.classList.add(
                "hidden"
            );
        }

        if (verifyTab) {
            verifyTab.classList.add(
                "active"
            );
        }

        if (enrollTab) {
            enrollTab.classList.remove(
                "active"
            );
        }
    }
    else {

        if (verifyPage) {
            verifyPage.classList.add(
                "hidden"
            );
        }

        if (enrollPage) {
            enrollPage.classList.remove(
                "hidden"
            );
        }

        if (verifyTab) {
            verifyTab.classList.remove(
                "active"
            );
        }

        if (enrollTab) {
            enrollTab.classList.add(
                "active"
            );
        }


        // -------------------------------------------------
        // Enrollment canvas may have been hidden.
        // Resize after displaying the page.
        // -------------------------------------------------

        setTimeout(
            function () {

                resizeEnrollmentECGCanvas();

                resizeECGCanvas();
            },
            50
        );
    }


    resetSystem();
}


// =========================================================
// Verification
// =========================================================

async function startVerification() {

    setButton(
        "verifyButton",
        true
    );

    hideResult(
        "verify"
    );

    hideDetectedUser();

    clearECGCanvas();

    clearEnrollmentECGCanvas();


    setSystem(
        "در حال بررسی اتصال...",
        true
    );

    setLive(
        "CHECKING",
        true
    );

    updateStep(1);

    setProcessingStatus(
        "بررسی"
    );


    try {

        /*
         * در احراز هویت username ارسال نمی‌شود.
         *
         * سیستم خودش ECG را با تمام Templateهای
         * موجود در user_database مقایسه می‌کند.
         */

        const response =
            await fetch(
                "/api/verify",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({})
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            showResult(
                "verify",
                "denied",
                "",
                "اتصال برقرار نیست",
                data.message ||
                "امکان شروع احراز هویت وجود ندارد."
            );


            setSystem(
                "اتصال برقرار نیست",
                false
            );

            setLive(
                "OFFLINE",
                false
            );

            setButton(
                "verifyButton",
                false
            );

            return;
        }


        startPolling();
    }


    catch (error) {

        console.error(
            "Verification error:",
            error
        );


        showResult(
            "verify",
            "denied",
            "",
            "خطای ارتباط",
            "ارتباط با سرور برقرار نشد."
        );


        setSystem(
            "خطای ارتباط با سرور",
            false
        );

        setLive(
            "ERROR",
            false
        );

        setButton(
            "verifyButton",
            false
        );
    }
}


// =========================================================
// Enrollment
// =========================================================

async function startEnrollment() {

    const input =
        document.getElementById(
            "enrollUsername"
        );


    if (!input) {
        return;
    }


    const username =
        input.value.trim();


    if (!username) {

        showResult(
            "enroll",
            "denied",
            "",
            "نام کاربر وارد نشده",
            "لطفاً نام کاربر جدید را وارد کنید."
        );

        return;
    }


    setButton(
        "enrollButton",
        true
    );


    hideResult(
        "enroll"
    );


    // Clear both possible monitors

    clearECGCanvas();

    clearEnrollmentECGCanvas();


    // Make sure enrollment monitor has correct size

    setTimeout(
        function () {
            resizeEnrollmentECGCanvas();
        },
        50
    );


    setSystem(
        "در حال بررسی اتصال...",
        true
    );

    setLive(
        "CHECKING",
        true
    );

    updateStep(1);


    try {

        const response =
            await fetch(
                "/api/enroll",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            username:
                                username
                        })
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            showResult(
                "enroll",
                "denied",
                "",
                "ثبت کاربر انجام نشد",
                data.message ||
                "امکان شروع ثبت کاربر وجود ندارد."
            );


            setSystem(
                "ثبت کاربر انجام نشد",
                false
            );

            setLive(
                "ERROR",
                false
            );

            setButton(
                "enrollButton",
                false
            );

            return;
        }


        startPolling();
    }


    catch (error) {

        console.error(
            "Enrollment error:",
            error
        );


        showResult(
            "enroll",
            "denied",
            "",
            "خطای ارتباط",
            "ارتباط با سرور برقرار نشد."
        );


        setSystem(
            "خطای ارتباط با سرور",
            false
        );

        setLive(
            "ERROR",
            false
        );

        setButton(
            "enrollButton",
            false
        );
    }
}


// =========================================================
// Polling
// =========================================================

function startPolling() {

    if (polling) {

        clearInterval(
            polling
        );
    }


    // Immediately check once

    checkStatus();


    polling =
        setInterval(
            checkStatus,
            ECG_POLL_INTERVAL
        );
}


// =========================================================
// Status
// =========================================================

async function checkStatus() {

    try {

        const response =
            await fetch(
                "/api/status",
                {
                    cache:
                        "no-store"
                }
            );


        const data =
            await response.json();


        // -------------------------------------------------
        // ECG MUST be updated on every status request.
        // This makes live ECG available for both modes.
        // -------------------------------------------------

        updateECGFromStatus(
            data
        );


        updateStatus(
            data
        );


        // -------------------------------------------------
        // Terminal states
        // -------------------------------------------------

        if (

            data.status ===
                "success" ||

            data.status ===
                "denied" ||

            data.status ===
                "error" ||

            data.status ===
                "finished" ||

            data.status ===
                "enroll_success" ||

            data.status ===
                "connection_error"

        ) {

            if (polling) {

                clearInterval(
                    polling
                );

                polling = null;
            }


            setButton(
                "verifyButton",
                false
            );

            setButton(
                "enrollButton",
                false
            );
        }
    }


    catch (error) {

        console.error(
            "Status error:",
            error
        );
    }
}


// =========================================================
// Update Status
// =========================================================

function updateStatus(data) {

    if (!data) {
        return;
    }


    // =====================================================
    // Connection Error
    // =====================================================

    if (
        data.status ===
        "connection_error"
    ) {

        setSystem(
            "اتصال برقرار نیست",
            false
        );

        setLive(
            "OFFLINE",
            false
        );

        setConnectionStatus(
            "قطع"
        );

        setProcessingStatus(
            "خطا"
        );


        showResult(
            data.mode || "verify",
            "denied",
            "",
            "اتصال برقرار نیست",
            data.message ||
            "اتصال Arduino یا MAX30003 برقرار نیست."
        );


        showECGOverlay(
            "",
            "اتصال برقرار نیست"
        );

        return;
    }


    // =====================================================
    // Recording
    // =====================================================

    if (
        data.status ===
        "recording"
    ) {

        setSystem(
            "در حال ضبط ECG...",
            true
        );

        setLive(
            "LIVE",
            true
        );

        setConnectionStatus(
            "متصل"
        );

        setProcessingStatus(
            "در حال ضبط"
        );

        updateStep(2);


        showECGOverlay(
            "",
            "در حال دریافت سیگنال ECG..."
        );

        return;
    }


    // =====================================================
    // Processing
    // =====================================================

    if (
        data.status ===
        "processing"
    ) {

        setSystem(
            "در حال پردازش...",
            true
        );

        setLive(
            "PROCESSING",
            true
        );

        setConnectionStatus(
            "متصل"
        );

        setProcessingStatus(
            "پردازش"
        );

        updateStep(3);


        showECGOverlay(
            "",
            "در حال پردازش سیگنال..."
        );

        return;
    }


    // =====================================================
    // Analysis
    // =====================================================

    if (
        data.status ===
        "analysis"
    ) {

        setSystem(
            "در حال تحلیل بیومتریک...",
            true
        );

        setLive(
            "ANALYSIS",
            true
        );

        setProcessingStatus(
            "CNN"
        );

        updateStep(3);


        showECGOverlay(
            "",
            "در حال تحلیل ویژگی‌های ECG..."
        );

        return;
    }


    // =====================================================
    // Verification Success
    // =====================================================

    if (
        data.status ===
        "success"
    ) {

        setSystem(
            "قفل باز شد",
            true
        );

        setLive(
            "UNLOCKED",
            true
        );

        setConnectionStatus(
            "متصل"
        );

        setProcessingStatus(
            "تأیید شد"
        );

        updateStep(4);


        // -------------------------------------------------
        // Show identified user
        // -------------------------------------------------

        if (data.username) {

            showDetectedUser(
                data.username
            );
        }


        // -------------------------------------------------
        // Final result
        // -------------------------------------------------

        showResult(
            "verify",
            "success",
            "✓",
            "قفل باز شد",
            data.message ||
            (
                data.username
                    ? `کاربر «${data.username}» تأیید شد — قفل باز شد.`
                    : "هویت کاربر تأیید شد."
            )
        );


        showECGOverlay(
            "",
            "دسترسی مجاز"
        );

        return;
    }


    // =====================================================
    // Enrollment Success
    // =====================================================

    if (
        data.status ===
        "enroll_success"
    ) {

        setSystem(
            "ثبت کاربر موفق بود",
            true
        );

        setLive(
            "SAVED",
            true
        );

        setConnectionStatus(
            "متصل"
        );

        setProcessingStatus(
            "ذخیره شد"
        );

        updateStep(4);


        showResult(
            "enroll",
            "success",
            "✓",
            "کاربر ثبت شد",
            data.message ||
            "Template کاربر با موفقیت ذخیره شد."
        );


        showECGOverlay(
            "",
            "Template با موفقیت ذخیره شد"
        );

        return;
    }


    // =====================================================
    // Verification Denied
    // =====================================================

    if (
        data.status ===
        "denied"
    ) {

        setSystem(
            "هویت تأیید نشد",
            false
        );

        setLive(
            "DENIED",
            false
        );

        setConnectionStatus(
            "متصل"
        );

        setProcessingStatus(
            "رد شد"
        );

        updateStep(4);

        hideDetectedUser();


        showResult(
            "verify",
            "denied",
            "",
            "هویت تأیید نشد",
            data.message ||
            "هیچ‌یک از کاربران موجود با سیگنال ECG مطابقت نداشت."
        );


        showECGOverlay(
            "",
            "هویت شناسایی نشد"
        );

        return;
    }


    // =====================================================
    // General Error
    // =====================================================

    if (
        data.status ===
        "error"
    ) {

        setSystem(
            "خطا در سیستم",
            false
        );

        setLive(
            "ERROR",
            false
        );

        setProcessingStatus(
            "خطا"
        );


        showResult(
            data.mode || "verify",
            "denied",
            "",
            "خطا در سیستم",
            data.message ||
            "در پردازش اطلاعات خطایی رخ داد."
        );


        showECGOverlay(
            "",
            "خطا در پردازش"
        );

        return;
    }
}


// =========================================================
// Result
// =========================================================

function showResult(
    mode,
    type,
    icon,
    title,
    message
) {

    let box;
    let iconElement;
    let titleElement;
    let messageElement;


    if (
        mode === "enroll"
    ) {

        box =
            document.getElementById(
                "enrollResult"
            );

        iconElement =
            document.getElementById(
                "enrollResultIcon"
            );

        titleElement =
            document.getElementById(
                "enrollResultTitle"
            );

        messageElement =
            document.getElementById(
                "enrollResultMessage"
            );
    }
    else {

        box =
            document.getElementById(
                "verifyResult"
            );

        iconElement =
            document.getElementById(
                "verifyResultIcon"
            );

        titleElement =
            document.getElementById(
                "verifyResultTitle"
            );

        messageElement =
            document.getElementById(
                "verifyResultMessage"
            );
    }


    if (!box) {
        return;
    }


    // Remove previous states

    box.classList.remove(
        "hidden",
        "success",
        "denied"
    );


    // Add current state

    box.classList.add(
        type
    );


    // -----------------------------------------------------
    // Icon
    // -----------------------------------------------------

    if (iconElement) {

        iconElement.textContent =
            icon || "";


        // Hide empty icon completely

        if (!icon) {

            iconElement.style.display =
                "none";
        }
        else {

            iconElement.style.display =
                "";
        }
    }


    // -----------------------------------------------------
    // Title
    // -----------------------------------------------------

    if (titleElement) {

        titleElement.textContent =
            title;
    }


    // -----------------------------------------------------
    // Message
    // -----------------------------------------------------

    if (messageElement) {

        messageElement.textContent =
            message;
    }
}


// =========================================================
// Hide Result
// =========================================================

function hideResult(mode) {

    const id =
        mode === "enroll"
            ? "enrollResult"
            : "verifyResult";


    const element =
        document.getElementById(id);


    if (element) {

        element.classList.add(
            "hidden"
        );
    }
}


// =========================================================
// Detected User
// =========================================================

function showDetectedUser(
    username
) {

    const container =
        document.getElementById(
            "detectedUser"
        );

    const name =
        document.getElementById(
            "detectedUsername"
        );


    if (
        !container ||
        !name
    ) {
        return;
    }


    name.textContent =
        username;


    container.classList.remove(
        "hidden"
    );
}


function hideDetectedUser() {

    const container =
        document.getElementById(
            "detectedUser"
        );


    if (container) {

        container.classList.add(
            "hidden"
        );
    }
}


// =========================================================
// Button
// =========================================================

function setButton(
    id,
    disabled
) {

    const button =
        document.getElementById(id);


    if (!button) {
        return;
    }


    button.disabled =
        disabled;


    if (disabled) {

        button.classList.add(
            "loading"
        );
    }
    else {

        button.classList.remove(
            "loading"
        );
    }
}


// =========================================================
// System Status
// =========================================================

function setSystem(
    text,
    active
) {

    const status =
        document.getElementById(
            "systemStatus"
        );

    const dot =
        document.getElementById(
            "systemDot"
        );


    if (status) {

        status.textContent =
            text;
    }


    if (!dot) {
        return;
    }


    if (active) {

        dot.style.background =
            "#10C98A";

        dot.style.boxShadow =
            "0 0 12px rgba(16,201,138,.8)";
    }
    else {

        dot.style.background =
            "#ef4444";

        dot.style.boxShadow =
            "0 0 12px rgba(239,68,68,.7)";
    }
}


// =========================================================
// Live Status
// =========================================================

function setLive(
    text,
    active
) {

    const liveText =
        document.getElementById(
            "liveText"
        );

    const dot =
        document.getElementById(
            "liveDot"
        );


    if (liveText) {

        liveText.textContent =
            text;
    }


    if (!dot) {
        return;
    }


    if (active) {

        dot.classList.add(
            "active"
        );
    }
    else {

        dot.classList.remove(
            "active"
        );
    }


    // Update enroll live badge if exists
    const enrollLiveText =
        document.getElementById(
            "enrollLiveText"
        );

    const enrollLiveDot =
        document.getElementById(
            "enrollLiveDot"
        );


    if (enrollLiveText) {

        enrollLiveText.textContent =
            text;
    }


    if (enrollLiveDot) {

        if (active) {

            enrollLiveDot.classList.add(
                "active"
            );
        }
        else {

            enrollLiveDot.classList.remove(
                "active"
            );
        }
    }
}


// =========================================================
// Connection Status
// =========================================================

function setConnectionStatus(
    text
) {

    const element =
        document.getElementById(
            "connectionStatus"
        );


    if (element) {

        element.textContent =
            text;
    }
}


// =========================================================
// Processing Status
// =========================================================

function setProcessingStatus(
    text
) {

    const element =
        document.getElementById(
            "processingStatus"
        );


    if (element) {

        element.textContent =
            text;
    }
}


// =========================================================
// ECG Overlay
// =========================================================

function showECGOverlay(
    icon,
    text
) {

    const overlay =
        document.getElementById(
            "ecgOverlay"
        );

    const overlayIcon =
        document.getElementById(
            "ecgOverlayIcon"
        );

    const overlayText =
        document.getElementById(
            "ecgOverlayText"
    );


    if (!overlay) {
        return;
    }


    // -----------------------------------------------------
    // Icon
    // -----------------------------------------------------

    if (overlayIcon) {

        overlayIcon.textContent =
            icon || "";


        if (!icon) {

            overlayIcon.style.display =
                "none";
        }
        else {

            overlayIcon.style.display =
                "";
        }
    }


    // -----------------------------------------------------
    // Text
    // -----------------------------------------------------

    if (overlayText) {

        overlayText.textContent =
            text;
    }


    // -----------------------------------------------------
    // Live state
    // -----------------------------------------------------

    if (
        text.includes("دریافت") ||
        text.includes("ضبط") ||
        text.includes("پردازش") ||
        text.includes("تحلیل") ||
        text.includes("دسترسی")
    ) {
        overlay.classList.add("live");
    }
    else {
        overlay.classList.remove("live");
    }


    // -----------------------------------------------------
    // Update enrollment overlay if exists
    // -----------------------------------------------------

    const enrollOverlay =
        document.getElementById(
            "enrollEcgOverlay"
        );

    const enrollOverlayText =
        document.getElementById(
            "enrollEcgOverlayText"
        );


    if (enrollOverlay && enrollOverlayText) {

        enrollOverlayText.textContent =
            text;


        if (
            text.includes("دریافت") ||
            text.includes("ضبط") ||
            text.includes("پردازش") ||
            text.includes("تحلیل") ||
            text.includes("دسترسی")
        ) {
            enrollOverlay.classList.add("live");
        }
        else {
            enrollOverlay.classList.remove("live");
        }
    }
}


// =========================================================
// Steps
// =========================================================

function updateStep(
    number
) {

    for (
        let i = 1;
        i <= 4;
        i++
    ) {

        const step =
            document.getElementById(
                `step${i}`
            );


        if (!step) {
            continue;
        }


        if (i <= number) {

            step.classList.add(
                "active"
            );
        }
        else {

            step.classList.remove(
                "active"
            );
        }
    }
}


// =========================================================
// Reset
// =========================================================

async function resetSystem() {

    // -----------------------------------------------------
    // Stop polling
    // -----------------------------------------------------

    if (polling) {

        clearInterval(
            polling
        );

        polling = null;
    }


    // -----------------------------------------------------
    // Reset backend
    // -----------------------------------------------------

    try {

        await fetch(
            "/api/reset",
            {
                method: "POST"
            }
        );
    }


    catch (error) {

        console.log(
            "Reset error:",
            error
        );
    }


    // -----------------------------------------------------
    // Reset ECG
    // -----------------------------------------------------

    clearECGCanvas();

    clearEnrollmentECGCanvas();


    // -----------------------------------------------------
    // Reset UI
    // -----------------------------------------------------

    setSystem(
        "سیستم آماده است",
        true
    );

    setLive(
        "آماده",
        false
    );

    setConnectionStatus(
        "آماده"
    );

    setProcessingStatus(
        "انتظار"
    );


    updateStep(1);

    hideDetectedUser();


    showECGOverlay(
        "",
        "آماده دریافت سیگنال"
    );
}


// =========================================================
// Enter Key - Enrollment
// =========================================================

function initializeEnrollmentInput() {

    const enrollUsername =
        document.getElementById(
            "enrollUsername"
        );


    if (!enrollUsername) {
        return;
    }


    enrollUsername.addEventListener(
        "keydown",
        function (event) {

            if (
                event.key === "Enter"
            ) {

                event.preventDefault();

                startEnrollment();
            }
        }
    );
}


// =========================================================
// DOM Loaded
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        // -------------------------------------------------
        // Enrollment input
        // -------------------------------------------------

        initializeEnrollmentInput();


        // -------------------------------------------------
        // Initialize ECG
        // -------------------------------------------------

        initializeECG();


        // -------------------------------------------------
        // Initial state
        // -------------------------------------------------

        setSystem(
            "سیستم آماده است",
            true
        );

        setLive(
            "آماده",
            false
        );

        setConnectionStatus(
            "آماده"
        );

        setProcessingStatus(
            "انتظار"
        );

        updateStep(1);


        showECGOverlay(
            "",
            "آماده دریافت سیگنال"
        );
    }
);